import logging
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances, PolygonMasks
from detectron2.utils.events import get_event_storage
import cv2
import os
from .fast_rcnn import FastRCNNOutputs
from .eopsn_inference import  eopsn_inference
from torchvision.ops import nms
import util.misc as utils
from util.clustering import clustering
import torch.distributed as dist
import numpy as np
import glob

__all__ = ["FastRCNNOutputLayers_eopsn"]


logger = logging.getLogger(__name__)

def pad(input, length=20, value=-1234):
    if len(input.shape) >= 1:
        return torch.cat((input, value * torch.ones([length-len(input)] + list(input.shape[1:]), device=input.device)))
    return torch.cat((input, value * torch.ones((length-len(input),), device=input.device)))

def size_condition(area, option='lm'):
    if option == 'lm':
        return area > 32 ** 2
    elif option == 'm':
        return (area > 32**2) & (area <= 96**2)
    elif option == 'l':
        return area > 96 ** 2
    elif option == 's':
        return area <= 32 ** 2
    elif option == 'ms':
        return area <= 96 ** 2
    return area > 0

def get_cos_sim(x, y):
    return torch.matmul(x.div(x.norm(dim=-1, p=2, keepdim=True) + 1e-6),
                        y.div(y.norm(dim=-1, p=2, keepdim=True) + 1e-6).T)


def do_nms(proposals, image_path, flips=None, nms_thresh=0.1, size_opt='lm'):
    idx = []
    bbox = []
    paths = []
    l = 0
    for i, (x, path) in enumerate(zip(proposals, image_path)):
        path = int(path.split('/')[-1].split('.')[0])
        boxes = x.proposal_boxes.tensor
        H, W = x._image_size

        area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        ind = size_condition(area, size_opt)
        boxes = boxes[ind]
        logits = x.objectness_logits[ind]
        nonzero_ind = torch.nonzero(ind).view(-1)

        if hasattr(x, 'gt_classes'):
            ind = x.gt_classes[ind] == -1
            boxes = boxes[ind]
            logits = logits[ind]
            nonzero_ind = nonzero_ind[ind]

        keep = nms(boxes, logits, nms_thresh)
        nonzero_ind = nonzero_ind[keep]
        idx.append(l + nonzero_ind)
        l  = len(x)
        paths.append(torch.ones((l), device=logits.device)*path)
        boxes = boxes.div(torch.as_tensor([[W, H, W, H]], device=boxes.device))
        if flips[i] == 1:
            boxes[:,0] = 1-boxes[:,0]
            boxes[:,2] = 1-boxes[:,2]
            boxes = torch.index_select(boxes,-1, torch.as_tensor([2,1,0,3], device=boxes.device))

        bbox.append(boxes)

    return torch.cat(idx), torch.cat(bbox), torch.cat(paths)



class FastRCNNOutputLayers_eopsn(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        test_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        box_reg_loss_weight=1.0,
        add_unlabeled_class=False,
        label_converter=None,
        reverse_label_converter=None,
        num_centroid=256,
        clustering_interval=1000,
        cluster_obj_thresh=0.8,
        coupled_cos_thresh=0.15,
        coupled_obj_thresh=0.9,
        cos_thresh=0.15,
        pos_class_thresh=0.7,
        nms_thresh=0.3,
        n_sample=20,
        output_dir='./'
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            box_reg_loss_weight (float): Weight for box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.label_converter = label_converter
        self.reverse_label_converter = reverse_label_converter
        self.original_num_classes = len(self.label_converter)
        addition = self.label_converter.max() + torch.arange(num_centroid) + 1
        self.label_converter = torch.cat((self.label_converter, addition))

        if self.reverse_label_converter is not None:
            num_classes = min(num_classes+1, len(reverse_label_converter))
        num_cls = num_classes

        self.add_unlabeled_class = add_unlabeled_class
        self.num_classes = num_cls

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_cls - 1
        box_dim = len(box2box_transform.weights)
        self.cls_score = Linear(input_size, num_cls+num_centroid)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        self.box_reg_loss_weight = box_reg_loss_weight

        self.feature_memory = []
        self.label_memory = []
        self.obj_score_memory = []
        self.path_memory = []
        self.bbox_memory = []

        self.num_centroid = num_centroid
        self.clustering_interval = clustering_interval
        weight = torch.zeros((num_centroid, input_size))
        weight = torch.zeros((num_centroid, 1))
        weight = torch.zeros((num_centroid+num_cls, 1))
        weight[:num_cls] = 1
        self.cls_weight = nn.Embedding(num_centroid+num_cls,1).from_pretrained(weight, freeze=True)
        self.turn_on = False
        self.step = 1
        self.cluster_count = 1
        self.pseudo_gt = None
        self.n_pseudo_gt = 0

        self.n_sample = n_sample
        self.cluster_obj_thresh = cluster_obj_thresh
        self.cos_thresh = cos_thresh
        self.coupled_cos_thresh = coupled_cos_thresh
        self.coupled_obj_thresh = coupled_obj_thresh
        self.pos_class_thresh = pos_class_thresh
        self.nms_thresh = nms_thresh
        self.pal = np.random.random((1024,3)) * 255

        self.size_opt = 'lm'

        self.output_dir = output_dir

        g_list = glob.glob(os.path.join(self.output_dir, 'pseudo_gts', '*.pth'))
        if len(g_list) > 0:
            g_list = [int(x.split('/')[-1].replace('.pth','')) for x in g_list]
            g = max(g_list)
            path = os.path.join(self.output_dir,'pseudo_gts/{}.pth').format(g)
            self.pseudo_gt = torch.load(path)
            self.n_pseudo_gt = len(self.pseudo_gt)
            self.step = g + 1
            if self.pseudo_gt is not None and len(self.pseudo_gt) > 0:

                label = int(self.pseudo_gt[:,1].max())
                weight[:label] = 1
                self.cls_weight = nn.Embedding(num_centroid+num_cls,1).from_pretrained(weight, freeze=True)


    @classmethod
    def from_config(cls, cfg, input_shape, label_converter=None, reverse_label_converter=None):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "box_reg_loss_weight"   : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT,
            "add_unlabeled_class": cfg.MODEL.EOPSN.UNLABELED_REGION and (not cfg.MODEL.EOPSN.IGNORE_UNLABELED_REGION),
            "label_converter": label_converter,
            "reverse_label_converter": reverse_label_converter,
            "num_centroid": cfg.MODEL.EOPSN.NUM_CENTROID,
            "clustering_interval": cfg.MODEL.EOPSN.CLUSTERING_INTERVAL,
            "nms_thresh": cfg.MODEL.EOPSN.NMS_THRESH,
            "cluster_obj_thresh": cfg.MODEL.EOPSN.CLUSTER_OBJ_THRESH,
            "coupled_obj_thresh": cfg.MODEL.EOPSN.COUPLED_OBJ_THRESH,
            "cos_thresh": cfg.MODEL.EOPSN.COS_THRESH,
            "coupled_cos_thresh": cfg.MODEL.EOPSN.COUPLED_COS_THRESH,
            "output_dir": cfg.OUTPUT_DIR

            # fmt: on
        }

    def forward(self, x):
        """
        Returns:
            Tensor: shape (N,K+1), scores for each of the N box. Each row contains the scores for
                K object categories and 1 background class.
            Tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4), or (N,4)
                for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas, x


    def gather(self, input_list):
        input = torch.cat(input_list)
        size = utils.get_world_size()
        if size == 1:
            if len(input.shape) > 1:
                input = input[input.sum(-1) != -1234*input.shape[1]]
            else:
                input = input[input != -1234]
            return input
        input_list = [torch.zeros_like(input) for _ in range(size)]
        torch.cuda.synchronize()
        dist.all_gather(input_list, input)
        input = torch.cat(input_list)
        if len(input.shape) > 1:
            input = input[input.sum(-1) != -1234*input.shape[1]]
        else:
            input = input[input != -1234]
        return input

    def sync_pseudo_gt(self, templete=None, dir_name='pseudo_gts'):
        size = utils.get_world_size()
        if self.pseudo_gt is None or size == 1:
            return
        try:
            data = self.pseudo_gt[self.n_pseudo_gt:].view(-1,6)
            path = data[:, 0].long()
            label = data[:,1]
            boxes = data[:,2:]
            dir_name = os.path.join(self.output_dir, dir_name)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            for p in path.unique():
                img = cv2.imread(templete.format(p))
                img_h, img_w, _ = img.shape
                multiplier = torch.tensor(
                    [img_w, img_h, img_w, img_h],
                    dtype=torch.float32, device=data.device)
                idx = path==p
                bbox = boxes[idx]
                bbox = bbox * multiplier
                bbox = bbox.int().cpu().numpy()
                lbl = label[idx]
                if not os.path.exists(dir_name+'/{:05}'.format(self.step)):
                    os.mkdir(dir_name+'/{:05}'.format(self.step))
                for i, box in enumerate(bbox):
                    cropped_image = img[box[1]:box[3]+1, box[0]:box[2]+1]
                    framed_image = cropped_image.astype(np.uint8)
                    cropped_img_path = os.path.join(dir_name, '{:05}/{:03}_{:012}_{:03}.jpg'.format(self.step,int(lbl[i]), int(p), i))
                    out = cv2.imwrite(cropped_img_path, framed_image)
                    if not out:
                        print("FAIL TO SAVE")
        except:
            print("FAIL TO SAVE")

        rank = utils.get_rank()
        array = torch.zeros((size,1), device=self.pseudo_gt.device)
        array[rank] = len(self.pseudo_gt) - self.n_pseudo_gt
        dist.all_reduce(array, dist.ReduceOp.SUM)
        data = self.pseudo_gt[self.n_pseudo_gt:]
        max_size = int(array.max())
        data = torch.cat((data, torch.zeros((max_size-len(data), data.shape[1]), device=data.device)))
        input_list = [torch.empty(size=(max_size, 6), device=self.pseudo_gt.device) for i in array]
        dist.all_gather(input_list, data)
        input_list = [e[:int(array[i])] for i,e in enumerate(input_list)]
        data = torch.cat(input_list)
        print("{} data sync".format(len(data)))
        self.pseudo_gt = torch.cat((self.pseudo_gt, data))
        self.n_pseudo_gt = len(self.pseudo_gt)
        if utils.get_rank() == 0:
            print(array)
            torch.save(self.pseudo_gt.cpu(), os.path.join(self.output_dir,'pseudo_gts/{}.pth'.format(self.step)))


    def clustering(self, image_path=None):
        # sync data
        self.sync_pseudo_gt()
        feature = self.gather(self.feature_memory)
        obj_score = self.gather(self.obj_score_memory)
        paths = self.gather(self.path_memory)
        bbox = self.gather(self.bbox_memory)
        self.feature_memory = []
        self.obj_score_memory = []
        self.path_memory = []
        self.bbox_memory = []

        if utils.get_rank() == 0 and self.cls_weight.weight.sum() < len(self.cls_weight.weight):
            ids, centroid, var = clustering(feature, K=self.num_centroid,
                                           step=self.step,
                                           device=feature.device,
                                             tol=1e-3,
                                             Niter=150
                                            )
            count = torch.bincount(ids)
            mean_obj_score = torch.bincount(ids, weights=obj_score.to(ids.device)) / (count + 1e-6)

            # top 10 % dense clusters.
            dist_topk_bound = -torch.topk(-var.view(-1), k=min(len(mean_obj_score), 13)).values[-1]
            mask = var < dist_topk_bound

            # number of found unknown classes
            cls_weight = sum(self.cls_weight.weight) - self.num_classes

            # high objectness clusters.
            cluster_obj_thresh = min(self.cluster_obj_thresh *  (1 + cls_weight / len(self.cls_weight.weight)), 0.99)
            obj_mask = mean_obj_score.to(mask.device) > cluster_obj_thresh

            mask = torch.logical_and(mask, obj_mask.to(mask.device))
            mask = mask.bool().view(-1)
            ids = ids.long().view(-1)

            paths = paths[mask[ids]]
            bbox = bbox[mask[ids]]
            feature = feature[mask[ids]]
            obj_score = obj_score[mask[ids]]
            ids = ids[mask[ids]]

            centroid = centroid[mask]

            if len(obj_score) > 0:
                obj_thresh = min(self.coupled_obj_thresh, max(obj_score))
            else:
                obj_thresh = self.coupled_obj_thresh

            obj_thresh = obj_thresh + (self.n_pseudo_gt * 0.01 / 100)
            obj_thresh = min(obj_thresh, 0.99)
            idx = obj_score >= obj_thresh
            bbox = bbox[idx]
            feature = feature[idx]
            paths = paths[idx]
            obj_score = obj_score[idx]
            ids = ids[idx]

            feats = []
            boxes = []
            ps = []
            obj_scores = []
            new_ids = []
            cls_weight = sum(self.cls_weight.weight) - self.num_classes

            coupled_cos_thresh = self.coupled_cos_thresh * (1 - cls_weight / len(self.cls_weight.weight))
            coupled_cos_thresh = max(coupled_cos_thresh, 0.01)
            for i, l in enumerate(sorted(ids.unique())):
                idx = ids == l
                feat = feature[idx]
                bb = bbox[idx]
                path = paths[idx]
                obj = obj_score[idx]

                cos_sim = get_cos_sim(feat, feat).view(-1)
                cos_dist = 1 - cos_sim
                idx = cos_dist.argsort()

                used = []
                used_path = []
                printer = cos_sim[idx]
                printer = printer[printer < 0.99999] # eliminate same element pairs
                for v in idx:
                    x, y = v // len(feat), v % len(feat)
                    if cos_dist[v] > coupled_cos_thresh:
                        break

                    if path[x] != path[y] and path[x] not in used_path and path[y] not in used_path:
                        used.append(x);  used.append(y)
                        used_path.append(path[x]); used_path.append(path[y])

                if len(used) > 0:
                    idx = torch.as_tensor(used, device=feat.device)
                    temp_ids = torch.ones((len(used),), device=feat.device) * l
                    feats.append(feat[idx])
                    boxes.append(bb[idx])
                    ps.append(path[idx])
                    obj_scores.append(obj[idx])
                    new_ids.append(temp_ids)
            if len(feats) > 0:
                feature = torch.cat(feats)
                bbox = torch.cat(boxes)
                paths = torch.cat(ps)
                obj_score = torch.cat(obj_scores)
                ids = torch.cat(new_ids)
                cls_weight = self.cls_weight.weight
                start_l = int(cls_weight.sum()) + self.original_num_classes - self.num_classes
                labels = -ids - 1
                unique_label = labels.unique()
                unique_label = unique_label[:cls_weight.shape[0]-int(cls_weight.sum())]
                for i, p in enumerate(unique_label):
                    if i + start_l - self.original_num_classes == self.num_centroid:
                        break
                    labels[labels==p] = i + start_l

                idx = labels > 0
                obj_score = obj_score[idx]
                labels = labels[idx]
                paths = paths[idx]
                feature = feature[idx]
                bbox = bbox[idx]

                data = torch.cat((paths.unsqueeze(1), labels.unsqueeze(1).float(), bbox), dim=-1)
            else:
                data = torch.zeros((0,6),device=feature.device)
            if image_path is not None and len(data) > 0:
                utils.save_boxes(data, feature.detach(), obj_score.detach(), image_path, self.pal, self.step, self.num_classes, self.output_dir)
            size = torch.as_tensor([len(data), len(centroid)], device=feature.device).float()

            storage = get_event_storage()
            storage.put_scalar("exemplar/obj_th", float(obj_thresh))
            storage.put_scalar("exemplar/cluster_obj_th", float(cluster_obj_thresh))
            storage.put_scalar("exemplar/sel_cluster", int(mask.sum()))
            storage.put_scalar("exemplar/coupled_cos_th", float(coupled_cos_thresh))
            storage.put_scalar("exemplar/new", len(data))
        else:
            size = torch.empty(size=(1,2), device=feature.device)

        # gather
        if utils.get_world_size() > 1:
            torch.cuda.synchronize()
            dist.broadcast(size, 0)
            if utils.get_rank() > 0:
                data = torch.empty(size=(int(size[0,0]), 6), device=feature.device)
            torch.cuda.synchronize()
            dist.broadcast(data, 0)
        l_cls = self.original_num_classes - 1
        l_new = int(data[:,1].max() - l_cls) if len(data) > 0 else 0

        cls_weight = self.cls_weight.weight.data
        cls_weight[:self.num_classes+l_new] = 1
        self.cls_weight.weight.data = cls_weight

        if self.pseudo_gt is None:
            self.pseudo_gt = data
        else:
            self.pseudo_gt = torch.cat((self.pseudo_gt, data))
        self.n_pseudo_gt = len(self.pseudo_gt)

        # flush
        if utils.get_rank() == 0:
            try:
                torch.save(self.pseudo_gt.cpu(), os.path.join(self.output_dir,'pseudo_gts/{}.pth'.format(self.step)))
            except:
                pass

    def add_feature(self, predictions, proposals, void_predictions, void_proposals, image_path=None, flips=None):
        void_scores, _, void_feature = void_predictions
        void_objectness_scores = torch.cat([x.objectness_logits.sigmoid() for x in void_proposals])
        weight = self.cls_weight.weight
        if weight.sum() < len(weight):
            if len(void_feature) > 0:
                nms_idx, bbox, paths = do_nms(void_proposals, image_path, flips, self.nms_thresh, self.size_opt)
                void_feature = void_feature[nms_idx]
                void_objectness_scores = void_objectness_scores[nms_idx]

            # void

            if len(void_objectness_scores) > 0:
                try:
                    idx = list(WeightedRandomSampler(void_objectness_scores, min(len(void_objectness_scores), self.n_sample), replacement=False))
                    feat = void_feature[idx].detach()
                    feat = torch.cat((feat, -1234 * torch.ones((self.n_sample-len(idx), feat.shape[1]), device=feat.device)))
                    obj = void_objectness_scores[idx].detach()
                    obj = torch.cat((obj, -1234 * torch.ones((self.n_sample-len(idx),), device=feat.device)))

                    bbox = pad(bbox[idx].detach())
                    paths = pad(paths[idx])
                except: # in the case that WeightedRandomSampler does not work.
                    feat = - 1234 * torch.ones((self.n_sample, 1024), device=void_feature.device)
                    obj = - 1234 * torch.ones((self.n_sample,), device=void_feature.device)
                    bbox  = - 1234 * torch.ones((self.n_sample, 4), device=void_feature.device)
                    paths = - 1234 * torch.ones((self.n_sample,), device=void_feature.device)
            else:
                feat = - 1234 * torch.ones((self.n_sample, 1024), device=void_feature.device)
                obj = - 1234 * torch.ones((self.n_sample,), device=void_feature.device)
                bbox  = - 1234 * torch.ones((self.n_sample, 4), device=void_feature.device)
                paths = - 1234 * torch.ones((self.n_sample,), device=void_feature.device)

            self.feature_memory.append(feat)
            self.obj_score_memory.append(obj)
            self.path_memory.append(paths)
            self.bbox_memory.append(bbox)

            if self.step % self.clustering_interval == 0:
                self.clustering(image_path)

    def add_pseudo_label(self, targets, image_path, flip):
        new_targets = []
        if self.pseudo_gt is None:
            return targets
        if len(targets) > 0 and targets[0].gt_boxes.tensor.device != self.pseudo_gt.device:
            self.pseudo_gt = self.pseudo_gt.to(targets[0].gt_boxes.tensor.device)
        for i, (targets_per_image, path) in enumerate(zip(targets, image_path)):
            H, W = targets_per_image._image_size
            gt_boxes = targets_per_image.gt_boxes
            gt_classes = targets_per_image.gt_classes
            p = int(path.split('/')[-1].split('.')[0])
            data = self.pseudo_gt[self.pseudo_gt[:,0] == p]
            ld = len(data)
            if len(data) == 0:
                new_targets.append(targets_per_image)
                continue
            label = data[:,1].long()
            boxes = data[:,2:].clone()
            if flip[i] == 1:
                boxes[:,0] = 1 - boxes[:, 0]
                boxes[:,2] = 1 - boxes[:, 2]
                boxes = torch.index_select(boxes, -1, torch.as_tensor([2,1,0,3], device=boxes.device))
            boxes = Boxes(boxes)
            boxes.scale(scale_x=W, scale_y=H)
            new_gt_boxes = gt_boxes.cat([gt_boxes, boxes])

            new_gt_masks = PolygonMasks([[]])
            if hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks
                new_gt_masks = new_gt_masks.cat([gt_masks] + [new_gt_masks]*ld)
            else:
                new_gt_masks = new_gt_masks.cat([new_gt_masks]*ld)
            new_gt_classes = torch.cat((gt_classes, label))

            new_target = Instances((H, W))
            new_target.gt_classes = new_gt_classes
            new_target.gt_masks = new_gt_masks
            new_target.gt_boxes = new_gt_boxes
            new_targets.append(new_target)
            lbl, cnt = label.unique(return_counts=True)
        return new_targets


    def add_exemplar(self, exemplar_info, void_features, void_proposals, image_path, flips,  dir_name='pseudo_gts'):
        exemplar_features, exemplar_labels, exemplar_length = exemplar_info

        p = image_path[0].split('/')[-1].split('.')[0]
        templete = image_path[0].replace(p,'{:012}')
        if self.step % 100 == 0: # sync multi-gpus
            self.sync_pseudo_gt(templete, dir_name)
        if len(exemplar_features) ==0 or len(void_features) ==0 :
            if utils.get_rank() == 0:
                storage = get_event_storage()
                storage.put_scalar("exemplar/add_exemplar", 0)
            return None

        boxes = [x.proposal_boxes.tensor for x in void_proposals]
        l = [len(b) for b in boxes]
        sizes = [x._image_size for x in void_proposals]
        cos = get_cos_sim(void_features, exemplar_features)
        th = max(0.01, self.cos_thresh - (0.01 * self.n_pseudo_gt / 200))
        if float(cos.max()) < 1 - th:
            if utils.get_rank() == 0:
                storage = get_event_storage()
                storage.put_scalar("exemplar/add_exemplar", 0)
            return None
        cos = cos.split(l)
        data = []
        cos_log = []
        label_log = []
        new_label = [-torch.ones((len(x),), device=cos[0].device) for x in void_proposals]

        for i, (c,  bbox, p, s) in enumerate(zip(cos, boxes, image_path, sizes)):
            H, W = s
            area = (bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])
            ind = size_condition(area, self.size_opt)
            bbox = bbox[ind]
            nonzero_ind = ind.nonzero()
            if len(bbox) == 0:
                continue
            c = c[ind]

            score, ind = c.view(len(bbox), -1).max(dim=0)
            bbox = bbox[ind]
            cc = score
            labels = exemplar_labels
            nonzero_ind = nonzero_ind[ind]
            ind = cc > 1-th

            cc = cc[ind]
            bbox = bbox[ind]
            nonzero_ind = nonzero_ind[ind]

            keep = nms(bbox, cc, self.nms_thresh)
            bbox = bbox[keep]
            cc = cc[keep]
            l = labels[keep]
            nonzero_ind = nonzero_ind[keep]
            bbox = bbox.div(torch.as_tensor([[W, H, W, H]], device=bbox.device))
            if flips[i] == 1:
                bbox[:,0] = 1 - bbox[:, 0]
                bbox[:,2] = 1 - bbox[:, 2]
                bbox = torch.index_select(bbox, -1, torch.as_tensor([2,1,0,3], device=bbox.device))
            labels = l.view(-1,1).float()
            new_label[i][nonzero_ind] = labels
            path = int(p.split('/')[-1].split('.')[0])
            pa = torch.ones((len(bbox),1), device=bbox.device) * path
            datum = torch.cat((pa, labels, bbox), dim=-1)
            data.append(datum)
            cos_log.append(cc)
            label_log.append(l)

        if len(data) > 0:
            dir_name = os.path.join(self.output_dir, dir_name)
            data = torch.cat(data)
            self.pseudo_gt = torch.cat((self.pseudo_gt, data))
        if utils.get_rank() == 0:
            storage = get_event_storage()
            storage.put_scalar("exemplar/add_exemplar", len(data))

        return new_label

    def losses(self, predictions, proposals, void_predictions, void_proposals, image_path=None, flips=None, use_exemplar=False):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        if utils.get_rank() == 0:
            storage = get_event_storage()
            storage.put_scalar("exemplar/num_pseudo_gt", len(self.pseudo_gt) if self.pseudo_gt is not None else 0)
        scores, proposal_deltas, feature = predictions
        void_scores, _, void_feature = void_predictions

        if len(void_scores) > 0:
            neg_sample = void_scores
            storage = get_event_storage()
            storage.put_scalar("exemplar/num_neg_sample", len(neg_sample))

            void_neg_loss = -torch.log(1-neg_sample.softmax(-1)[:, :self.num_classes-1]+1e-8)
            if len(void_neg_loss) > 0:
                void_neg_loss = void_neg_loss.sum() / len(void_neg_loss)
            else:
                void_neg_loss = void_neg_loss.sum()
        else:
            void_neg_loss = scores.sum() * 0

        void_loss = {'loss_void_neg': void_neg_loss}
        if use_exemplar:
            a, b, c = void_predictions
            l = sum([len(x) for x in void_proposals[:-1]])
            self.add_feature(predictions, proposals, (a[:l], b[:l], c[:l]), void_proposals[:-1], image_path[:-1], flips[:-1])
        else:
            self.add_feature(predictions, proposals, void_predictions, void_proposals, image_path, flips)

        frcnn_outputs = FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            self.box_reg_loss_weight,
            self.label_converter,
            add_unlabeled_class=self.add_unlabeled_class,
            cls_weight=self.cls_weight.weight.view(-1),
            bg_class_ind=self.num_classes-1
        )
        losses = frcnn_outputs.losses()
        self.step += 1
        losses.update(void_loss)
        return losses

    def inference(self, predictions, proposals, use_unknown=False):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """

        scores, proposal_deltas, x = predictions
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        objness_scores = [x.objectness_logits for x in proposals]
        image_shapes = [x.image_size for x in proposals]
        if hasattr(proposals[0], 'void_disc'):
            objness_scores = [x.void_disc for x in proposals]
        return eopsn_inference(
            boxes,
            scores,
            image_shapes,
            objness_scores,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            use_unknown,
            reverse_label_converter=self.reverse_label_converter,
            num_classes=self.cls_weight.weight.shape[0]  #self.cls_score.L.weight.shape[0]
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        proposal_deltas = predictions[1]
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        scores = predictions[0]
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        probs = torch.cat((probs[:,:self.num_classes-1], probs[:,-self.num_centroid:]), dim=1)

        return probs.split(num_inst_per_image, dim=0)
