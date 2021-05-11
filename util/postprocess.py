import torch
import pdb

def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_limit,
    instances_confidence_threshold,
    unknown_results=None,
    unknown_instances_confidence_threshold=0.1
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each is the contiguous semantic
            category id

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)

    # devide known and unknown instances

    current_segment_id = 0
    segments_info = []
    def _masking(results,
                 panoptic_seg, segments_info, current_segment_id, confidence_threshold, overlap_threshold,
                 add_id=0):


        masks = results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)
        scores = results.scores
        pred_classes = results.pred_classes

        sorted_inds = torch.argsort(-scores)
        for inst_id in sorted_inds:
            score = scores[inst_id].item()
            if score < confidence_threshold:
                break
            mask = masks[inst_id]  # H,W
            mask_area = mask.sum().item()

            if mask_area == 0:
                continue

            intersect = (mask > 0) & (panoptic_seg > 0)
            intersect_area = intersect.sum().item()

            if intersect_area * 1.0 / mask_area > overlap_threshold:
                continue

            if intersect_area > 0:
                mask = mask & (panoptic_seg == 0)

            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id
            c = pred_classes[inst_id].item()
            if c == -2:
                c = -1
            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": True,
                    "score": score,
                    "category_id": c,
                    "instance_id": inst_id.item() +  add_id ,
                }
            )
        return panoptic_seg, segments_info, current_segment_id

    unknown_instance_results = instance_results[instance_results.pred_classes ==-1]
    unknown_instance_results2 = instance_results[instance_results.pred_classes == -2]
    instance_results = instance_results[instance_results.pred_classes >= 0]

    panoptic_seg, segments_info, current_segment_id = _masking(instance_results,
                                                    panoptic_seg, segments_info, current_segment_id,
                                                    instances_confidence_threshold,
                                                    overlap_threshold)

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    for semantic_label in semantic_labels:
        if semantic_label == 0 or semantic_label == 54:  # 0 is a special "thing" class
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_limit:
            continue

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": False,
                "category_id": semantic_label,
                "area": mask_area,
            }
        )
    panoptic_seg, segments_info, current_segment_id = _masking(unknown_instance_results,
                                                    panoptic_seg, segments_info, current_segment_id,
                                                    unknown_instances_confidence_threshold,
                                                    overlap_threshold,
                                                    current_segment_id)
    panoptic_seg, segments_info, current_segment_id = _masking(unknown_instance_results2,
                                                    panoptic_seg, segments_info, current_segment_id,
                                                    unknown_instances_confidence_threshold,
                                                    overlap_threshold,
                                                    current_segment_id)


    return panoptic_seg, segments_info
