"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import torch.nn.functional as F
import pdb
import copy
import time

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def known_unknown_nms(known_boxes, unknown_boxes, thresh=0.5):
    iou, _ = box_iou(known_boxes, unknown_boxes)
    return (iou > thresh).sum(0) == 0

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def box_calibration(boxes, boundaries=None, masks=None, size=None):
    if masks is not None:
        boxes, masks = _box_calibration_mask(boxes, masks, size)
    if boundaries is not None:
        boxes = _box_calibration_boundary(boxes, boundaries)
        masks = None

    return boxes, masks


def _box_calibration_mask(boxes, masks, size):
    img_h, img_w = size
    boxes = box_cxcywh_to_xyxy(boxes)
    multiplier = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
    boxes = boxes * multiplier
    boxes = boxes.int()
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        dh = box[3] - box[1]
        dw = box[2] - box[0]
        conv_mask = F.interpolate(mask.view((1,1) + mask.shape), size=(dh, dw), mode='bilinear')
        th_mask = conv_mask > 0.5
        if th_mask.sum() == 0:
            continue
        mask_h = torch.nonzero(th_mask[0,0].sum(1) > 0)
        mask_w = torch.nonzero(th_mask[0,0].sum(0) > 0)
        h_st, h_ed = mask_h[0,0], mask_h[-1, 0]
        w_st, w_ed = mask_w[0,0], mask_w[-1, 0]
        conv_mask = conv_mask[:,:,h_st:h_ed+1, w_st:w_ed+1]
        h_ed = dh - 1 - h_ed
        w_ed = dw - 1 - w_ed
        box = box + torch.stack([h_st, w_st, h_ed, w_ed])
        mask = F.interpolate(conv_mask, size=mask.shape[-2:], mode='bilinear')[0,0]
        boxes[i] = box
        masks[i] = mask
    boxes = boxes / multiplier
    boxes = box_xyxy_to_cxcywh(boxes)
    return boxes, masks

def _box_calibration_boundary(boxes, boundaries):
    img_h, img_w = boundaries.shape[-2:]
    boxes = boxes.to(boundaries.device)
    boxes = box_cxcywh_to_xyxy(boxes)
    bbox_type = boxes.type()
    multiplier = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
    boxes = boxes * multiplier
    boxes = boxes.int()

    boundaries = boundaries[0,0]
    boxes = boxes.clamp(min=0)
    new_boxes = torch.zeros((0,4), device=boxes.device).long()
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        x2, y2 = x2.clamp(max=img_w), y2.clamp(max=img_h)
        area = boundaries[y1:y2, x1:x2]
        area_h, area_w = area.shape
        if area_h * area_w == 0:
            new_boxes = torch.cat((new_boxes, bbox.unsqueeze(0).long()))
            continue
        area_h, area_w = area_h // 2, area_w // 2
        if area_h > 0:
            hmax = area.max(dim=1).values
            top = hmax[:area_h].argmax()
            bottom = hmax[area_h:].argmax() + area_h
            yy1, yy2 = y1 + top, y2 + bottom
        else:
            yy1, yy2 = y1.long(), y2.long()

        if area_w > 0:
            wmax = area.max(dim=0).values
            left = wmax[:area_w].argmax()
            right = wmax[area_w:].argmax() + area_w
            xx1, xx2 = x1 + left, x1 + right
        else:
            xx1, xx2 = x1.long(), x2.long()
        bb = torch.cat((xx1.view(1,1), yy1.view(1,1), xx2.view(1,1), yy2.view(1,1)), dim=1)
        new_boxes = torch.cat((new_boxes, bb), dim=0)
    new_boxes = new_boxes.type(bbox_type)
    new_boxes = new_boxes / multiplier
    new_boxes = box_xyxy_to_cxcywh(new_boxes)
    return new_boxes



def add_unlabeled_class(boxes, target_classes, integral_sem_seg, ignore_value=255, bg=80):
#    B, C, H, W = one_hot.shape
    for b in range(len(boxes)):
        H, W = integral_sem_seg[b].shape
        one_hot = integral_sem_seg[b].view(-1)
        most_common_class = []
        idx = target_classes[b] == bg
        box = boxes[b][idx]
        box = box_cxcywh_to_xyxy(box) * torch.as_tensor([W, H, W, H], device=boxes.device,
                                                            dtype=torch.float32)
        box = box.long().clamp(min=0)
        lt = box[:, 1].clamp(max=H-1) * W + box[:, 0].clamp(max=W-1)
        lb = box[:, 1].clamp(max=H-1) * W + box[:, 2].clamp(max=W, min=1) - 1
        rt = (box[:,3].clamp(max=H, min=1)-1) * W + box[:, 0].clamp(max=W-1)
        rb = (box[:,3].clamp(max=H, min=1)-1)* W + box[:, 2].clamp(max=W, min=1) -1

        sel_lt = torch.index_select(one_hot, -1, lt)
        sel_lb = torch.index_select(one_hot, -1, lb)
        sel_rt = torch.index_select(one_hot, -1, rt)
        sel_rb = torch.index_select(one_hot, -1, rb)

        area = (rb - rt) * (rb - lb) / W
        c = (sel_rb - sel_rt + sel_lb - sel_lt)
        most_common_class = c < 0.5 * area
        temp = target_classes[b,idx]
        temp[most_common_class] += 1 # new bg
        target_classes[b,idx] = temp
    del one_hot
    return target_classes.long()
