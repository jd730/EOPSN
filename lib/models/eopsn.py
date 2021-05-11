# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import ImageList
from detectron2.data import MetadataCatalog
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.modeling import (PanopticFPN, META_ARCH_REGISTRY, build_backbone, detector_postprocess,
                                 build_roi_heads, build_mask_head, build_sem_seg_head)
from detectron2.modeling.postprocessing import sem_seg_postprocess

from util.postprocess import combine_semantic_and_instance_outputs

from lib.datasets import DatasetMapper

__all__ = ["EOPSN"]

@META_ARCH_REGISTRY.register()
class EOPSN(PanopticFPN):
    """
    Implement the paper :paper:`PanopticFPN`.
    """

    def __init__(self, cfg):
        unseen_path = cfg.DATASETS.UNSEEN_LABEL_SET
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        if unseen_path != '':
            meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            meta = {e: i for i, e in enumerate(meta)}
            with open(unseen_path, 'r') as f:
                lines = [meta[e.replace('\n','')] for e in f.readlines()]
            self.unseen_label_set = lines
            self.meta.stuff_classes.append('unknown')
            self.meta.stuff_colors.append([20, 220, 60])
            self.meta.stuff_dataset_id_to_contiguous_id[201] = 54
        else:
            self.unseen_label_set = None

        dataset = get_detection_dataset_dicts(
                cfg.DATASETS.TRAIN,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if cfg.MODEL.KEYPOINT_ON
                else 0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            )
        self.dataset = {int(e['file_name'].split('/')[-1].split('.')[0]): e for e in dataset}
        self.mapper = DatasetMapper(cfg, True)

        super().__init__(cfg)

        self.unlabeled_region_on = cfg.MODEL.EOPSN.UNLABELED_REGION
        self.ignore_unlabeled_region = cfg.MODEL.EOPSN.IGNORE_UNLABELED_REGION


    def get_exemplar_input(self, image_path, sample_size=1):
        pseudo_gt = self.roi_heads.box_predictor.pseudo_gt
        if pseudo_gt is None or len(pseudo_gt) == 0:
            return None

        ps = [int(e.split('/')[-1].split('.')[0]) for e in image_path]
        fnames = pseudo_gt[:,0].unique()
        ret = []
        for f in fnames[torch.randperm(fnames.size(0))]:
            fi = int(f)
            if fi not in ps and fi in self.dataset:
                ret.append(self.mapper.__call__(self.dataset[fi]))
            if len(ret) == sample_size:
                break
        return ret


    def _forward(self, batched_inputs):
        image_path = [x['file_name'] for x in batched_inputs]
        if self.training:
            flips = [x['flip'] for x in batched_inputs]
        else:
            flips = None
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)
        proposals = None

        if "proposals" in batched_inputs[0]:
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        if "sem_seg" in batched_inputs[0]:
            gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            gt_sem_seg = None
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)

        if "integral_sem_seg" in batched_inputs[0] and self.training:
            gt_integral_sem_seg = [x["integral_sem_seg"].to(self.device) for x in batched_inputs]
        else:
            gt_integral_sem_seg = None

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, gt_integral_sem_seg)
        else:
            proposal_losses = {}

        if "instances" in batched_inputs[0]:
            if hasattr(self.roi_heads.box_predictor, 'add_pseudo_label'):
                gt_instances = self.roi_heads.box_predictor.add_pseudo_label(gt_instances, image_path, flips)
        losses = {}
        if self.training:
            losses.update(sem_seg_losses)
            losses.update(proposal_losses)

        return images, features, proposals, gt_instances, gt_integral_sem_seg, sem_seg_results, losses


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                  See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """

        image_path = [x['file_name'] for x in batched_inputs]
        if self.training:
            flips = [x['flip'] for x in batched_inputs]
        else:
            flips = None

        if self.training:
            exemplar_input = self.get_exemplar_input(image_path, sample_size=1)
            if exemplar_input is not None:
                l = len(batched_inputs)
                batched_inputs = batched_inputs + exemplar_input
                images, features, proposals, gt_instances, gt_integral_sem_seg, _, losses = self._forward(batched_inputs)
                exemplar_features = {}
                for k, v in features.items():
                    exemplar_features[k] = v[l:]
                exemplar_gt_instances = gt_instances[l:]
                image_path = [x['file_name'] for x in batched_inputs]
                if self.training:
                    exemplar_flips = [x['flip'] for x in batched_inputs]
                else:
                    exemplar_flips = None
                with torch.no_grad():
                    exemplar_info = self.roi_heads.get_box_features(exemplar_features, exemplar_gt_instances)
                detector_results, detector_losses = self.roi_heads(
                    images, features, proposals, gt_instances, gt_integral_sem_seg,
                    image_path=image_path, flips=exemplar_flips, exemplar_info=exemplar_info
                )
                del exemplar_info, exemplar_input
            else:
                exemplar_info = None
                images, features, proposals, gt_instances, gt_integral_sem_seg, _, losses = self._forward(batched_inputs)
                detector_results, detector_losses = self.roi_heads(
                    images, features, proposals, gt_instances, gt_integral_sem_seg,
                    image_path=image_path, flips=flips, exemplar_info=exemplar_info
                )
        else:
            exemplar_info = None
            images, features, proposals, gt_instances, gt_integral_sem_seg, sem_seg_results, losses = self._forward(batched_inputs)
            detector_results, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances, gt_integral_sem_seg,
                image_path=image_path, flips=flips, exemplar_info=exemplar_info
            )

        if self.training:
            losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
            return losses

        processed_results = []
        for sem_seg_result, detector_result, input_per_image, image_size in zip(
            sem_seg_results, detector_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            detector_r = detector_postprocess(detector_result, height, width)

            processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results

