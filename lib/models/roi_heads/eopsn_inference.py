import logging
import torch
from detectron2.layers import batched_nms, nonzero_tuple
from detectron2.structures import Boxes, Instances

from torchvision.ops import nms

__all__ = ["eopsn_inference"]


logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def eopsn_inference(boxes, scores, image_shapes, objness_scores,
                        score_thresh, nms_thresh, topk_per_image,
                        use_unknown=False, num_classes=80, reverse_label_converter=None):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        eopsn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, objness_scores_per_image,
            score_thresh, nms_thresh, topk_per_image, use_unknown, num_classes,
            reverse_label_converter
        )
        for scores_per_image, boxes_per_image, image_shape, objness_scores_per_image in zip(scores, boxes, image_shapes, objness_scores)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def eopsn_inference_single_image(
    boxes, scores, image_shape, objness_scores, score_thresh, nms_thresh, topk_per_image,
    use_unknown=False, num_classes=80, reverse_label_converter=None
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)


    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        objness_scores = objness_scores[valid_mask]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K

    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    classes = filter_inds[:,-1]
    classes[classes > len(reverse_label_converter)-1] = -1
    filter_inds[:,-1] = classes

    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    classes = filter_inds[:,-1]
    if reverse_label_converter is not None:
        classes = reverse_label_converter.to(classes.device)[classes]

    boxes = boxes[:topk_per_image]
    scores = scores[:topk_per_image]
    classes = classes[:topk_per_image]

    inds = filter_inds[:,0]
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = classes
    inds = inds[:topk_per_image]
    return result, inds
