import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers_baseline
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads import ROIHeads, ROI_HEADS_REGISTRY
from detectron2.data import MetadataCatalog
from detectron2.utils.comm import all_gather

import util.misc as utils
from util.misc import add_unlabeled_class
from util.clustering import clustering
import torch.distributed as dist

logger = logging.getLogger('__name__')

def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes < bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads_EOPSN(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        ignore_unlabeled_region: bool = False,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.train_on_pred_boxes = train_on_pred_boxes
        self.ignore_unlabeled_region = ignore_unlabeled_region

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        ret["ignore_unlabeled_region"] = cfg.MODEL.EOPSN.IGNORE_UNLABELED_REGION
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))


        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        unseen_path = cfg.DATASETS.UNSEEN_LABEL_SET
        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        if unseen_path != '':
            meta_info = {e: i for i, e in enumerate(meta.thing_classes)}
            with open(unseen_path, 'r') as f:
                lines = [meta_info[e.replace('\n','')] for e in f.readlines()]
            unseen_label_set = sorted(lines)
            meta.stuff_classes.append('unknown')
            meta.stuff_colors.append([20, 220, 60])
            meta.stuff_dataset_id_to_contiguous_id[201] = 54
            if cfg.MODEL.EOPSN.IGNORE_UNLABELED_REGION or not cfg.MODEL.EOPSN.UNLABELED_REGION:
                label_converter = torch.ones(len(meta.thing_classes) + 1)
            else:
                label_converter = torch.ones(len(meta.thing_classes) + 2)
            for i in unseen_label_set:
                label_converter[i] = 0
            reverse_label_converter = label_converter.nonzero()[:,0].long()
            label_converter = torch.cumsum(label_converter, 0).long() - 1
            if cfg.MODEL.EOPSN.UNLABELED_REGION:
                if cfg.MODEL.EOPSN.IGNORE_UNLABELED_REGION:
                    reverse_label_converter[-1] = -1
                else:
                    reverse_label_converter[-1] = reverse_label_converter[-2]
                    reverse_label_converter[-2] = -1
        else:
            reverse_label_converter = None
            label_converter = None

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        if cfg.MODEL.EOPSN.PREDICTOR == 'baseline':
            box_predictor = FastRCNNOutputLayers_baseline(cfg, box_head.output_shape, label_converter, reverse_label_converter)
        elif cfg.MODEL.EOPSN.PREDICTOR == 'eopsn':
            from .eopsn_predictor import FastRCNNOutputLayers_eopsn
            box_predictor = FastRCNNOutputLayers_eopsn(cfg, box_head.output_shape, label_converter, reverse_label_converter)

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["mask_head"] = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )
        return ret

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], integral_sem_seg_target=None
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []
        void_proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for i, (proposals_per_image, targets_per_image) in enumerate(zip(proposals, targets)):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            if integral_sem_seg_target is not None:
                gt_classes[gt_classes==self.num_classes+1] = -730
                gt_classes, filtered_idx = add_unlabeled_class(proposals_per_image.proposal_boxes.tensor,
                                    gt_classes, integral_sem_seg_target[i], bg=self.num_classes)
                if self.ignore_unlabeled_region:
                    neg_filtered_idx = torch.logical_not(filtered_idx)
                    void_proposals_per_image = proposals_per_image[neg_filtered_idx]
                    gt_classes = gt_classes[filtered_idx]
                    gt_classes[gt_classes==1+self.num_classes] = self.num_classes
                    proposals_per_image = proposals_per_image[filtered_idx]
                else:
                    void_proposals_per_image = None
            gt_classes[gt_classes==-730] = self.num_classes+1
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                if self.ignore_unlabeled_region:
                    void_sampled_targets = sampled_targets[neg_filtered_idx]
                    sampled_targets = sampled_targets[filtered_idx]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
                        # maybe this is not needed
                        if self.ignore_unlabeled_region:
                            void_proposals_per_image.set(trg_name, trg_value[void_sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
                if void_proposals_per_image is not None:
                    gt_boxes = Boxes(
                        targets_per_image.gt_boxes.tensor.new_zeros((len(void_proposals_per_image), 4))
                    )
                    void_proposals_per_image.gt_boxes = gt_boxes


            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
            if self.ignore_unlabeled_region:
                void_proposals_with_gt.append(void_proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        if not self.ignore_unlabeled_region:
            return proposals_with_gt, None
        return proposals_with_gt, void_proposals_with_gt


    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        integral_sem_seg_target: Optional[List[torch.Tensor]] = None,
        image_path=None, flips=None,
        exemplar_info=None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals, void_proposals = self.label_and_sample_proposals(proposals, targets, integral_sem_seg_target)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, void_proposals, image_path=image_path, flips=flips, exemplar_info=exemplar_info)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            loss_masks = self._forward_mask(features, proposals)
            losses.update(loss_masks)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, image_path=image_path)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


    def get_box_features(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        features = [features[f].detach() for f in self.box_in_features]
        proposal_boxes = []
        exemplar_labels = []
        l = len(targets)
        for x in targets:
            idx = x.gt_classes > self.num_classes
            proposal_boxes.append(x.gt_boxes[idx])
            exemplar_labels.append(x.gt_classes[idx])
        box_features = self.box_pooler(features, proposal_boxes)
        box_features = self.box_head(box_features)
        del features
        return box_features, torch.cat(exemplar_labels), l


    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances], void_proposals: Optional[List[Instances]] = None,
        image_path=None, flips=None, exemplar_info=None
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        if self.training:
            void_box_features = self.box_pooler(features, [x.proposal_boxes for x in void_proposals])
            void_box_features = self.box_head(void_box_features)
            void_predictions = self.box_predictor(void_box_features)
            if exemplar_info is not None:
                with torch.no_grad():
                    ap = void_proposals[:-1]
                    l = sum([len(e) for e in ap])
                    lbl = self.box_predictor.add_exemplar(exemplar_info, void_box_features[:l].detach(), ap, image_path[:-1], flips[:-1])
                    if lbl is not None:
                        for x, l in zip(ap, lbl):
                            x.gt_classes = l
            del box_features
            losses = self.box_predictor.losses(predictions, proposals, void_predictions, void_proposals, image_path=image_path, flips=flips, use_exemplar=exemplar_info is not None)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, get_inds = self.box_predictor.inference(predictions, proposals, use_unknown=True)
            del box_features
            return pred_instances

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.mask_in_features]
        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)
