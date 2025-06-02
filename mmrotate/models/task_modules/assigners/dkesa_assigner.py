# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType
from mmengine.structures import InstanceData

from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class Point2RBoxAssigner(BaseAssigner):
    """Point2RBoxAssigner between the priors and gt boxes, which can achieve
    balance in positive priors, and gt_bboxes_ignore was not considered for
    now.

    Args:
        pos_ignore_thr (float): the threshold to ignore positive priors
        neg_ignore_thr (float): the threshold to ignore negative priors
        match_times(int): Number of positive priors for each gt box.
           Defaults to 4.
        iou_calculator (:obj:`ConfigDict` or dict): Config dict for iou
            calculator. Defaults to ``dict(type='BboxOverlaps2D')``
    """

    def __init__(
        self,
        pos_ignore_thr: float,
        neg_ignore_thr: float,
        match_times: int = 4,
        iou_calculator: ConfigType = dict(type='mmdet.BboxOverlaps2D')
    ) -> None:
        self.match_times = match_times
        self.pos_ignore_thr = pos_ignore_thr
        self.neg_ignore_thr = neg_ignore_thr
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def obb2xyxy(self, obb):
        w = obb[:, 2::5]
        h = obb[:, 3::5]
        a = obb[:, 4::5].detach()
        cosa = torch.cos(a).abs()
        sina = torch.sin(a).abs()
        hbbox_w = cosa * w + sina * h
        hbbox_h = sina * w + cosa * h
        dx = obb[..., 0]
        dy = obb[..., 1]
        dw = hbbox_w.reshape(-1)
        dh = hbbox_h.reshape(-1)
        x1 = dx - dw / 2
        y1 = dy - dh / 2
        x2 = dx + dw / 2
        y2 = dy + dh / 2
        return torch.stack((x1, y1, x2, y2), -1)

  

    def assign(
            self,
            pred_instances: InstanceData,
            gt_instances: InstanceData,
            gt_instances_ignore: Optional[InstanceData] = None
    ) -> AssignResult:
        """Assign gt to priors.

        The assignment is done in following steps

        1. assign -1 by default
        2. compute the L1 cost between boxes. Note that we use priors and
           predict boxes both
        3. compute the ignore indexes use gt_bboxes and predict boxes
        4. compute the ignore indexes of positive sample use priors and
           predict boxes


        Args:
            pred_instances (:obj:`InstaceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be priors, points, or bboxes predicted by the model,
                shape(n, 4).
            gt_instances (:obj:`InstaceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            gt_instances_ignore (:obj:`InstaceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """

        gt_bboxes = gt_instances.bboxes.tensor
        gt_labels = gt_instances.labels
        gt_bids = gt_instances.bids
        priors = pred_instances.priors
        bbox_pred = pred_instances.decoder_priors
        cls_scores = pred_instances.cls_scores

        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)


        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              0,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            assign_result = AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
            assign_result.set_extra_property(
                'pos_bbox_mask', bbox_pred.new_empty(0, dtype=torch.bool))
            assign_result.set_extra_property(
                'pos_point_mask', bbox_pred.new_empty(0, dtype=torch.bool))
            assign_result.set_extra_property(
                'pos_predicted_boxes',
                bbox_pred.new_empty((0, bbox_pred.shape[-1])))
            assign_result.set_extra_property(
                'target_boxes', gt_bboxes.new_empty((0, gt_bboxes.shape[-1])))
            assign_result.set_extra_property('target_labels',
                                             gt_labels.new_empty((0, )))
            assign_result.set_extra_property('target_bids',
                                             gt_bids.new_empty((0, )))
            return assign_result


        bbox_pred_xyxy = bbox_pred[:, :4]
        point_mask = gt_bboxes[:, 2] < 1
        gt_bboxes_xyxy = self.obb2xyxy(gt_bboxes)

        cost_center = torch.cdist(
            bbox_xyxy_to_cxcywh(bbox_pred_xyxy)[:, :2], gt_bboxes[:, :2], p=2)
        cost_cls_scores = 1 - cls_scores[:, gt_labels].sigmoid()
        cost_cls_scores[cost_center > 45.2] = 1e5

        cost_bbox = cost_cls_scores.clone()
        cost_bbox_priors = torch.cdist(
            bbox_xyxy_to_cxcywh(priors),
            bbox_xyxy_to_cxcywh(gt_bboxes_xyxy),
            p=1) * cost_cls_scores
        cost_bbox[:, point_mask] = 1e9
        cost_bbox_priors[:, point_mask] = 1e9

       
        cost_cls_scores[:, ~point_mask] = 1e9

        pairwise_cls=1-cost_cls_scores
        cls_cost = (
            F.binary_cross_entropy(
                valid_pred_scores.to(dtype=torch.float32).sqrt_(),
                gt_onehot_label,
                reduction='none',
            ).sum(-1).to(dtype=valid_pred_scores.dtype))

        cost_matrix = cls_cost
          

        matched_pred_ious, matched_gt_inds = \
            self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # is prior centers in gt centers
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (
            is_in_gts[is_in_gts_or_centers, :]
            & is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
