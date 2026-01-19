import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
import mmcv
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss, build_head
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from ...core.lane.util import fix_pts_interpolate


@HEADS.register_module()
class UniTopoHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=200,
                 num_lanes_one2one=200,
                 lanes_group=None,
                 loss_weight_group=1.0,
                 use_group_embed=False,
                 num_point=11,
                 query_embed_type='instance_pts',
                 transformer=None,
                 lclc_head=None,
                 lcte_head=None,
                 bbox_coder=None,
                 num_reg_fcs=2,
                 reg_embed_dims=None,
                 code_weights=None,
                 detach_te_feat=False,
                 bev_h=30,
                 bev_w=30,
                 pc_range=None,
                 pts_dim=3,
                 topo_head='toponet',
                 sync_cls_avg_factor=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.5),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.025),
                 train_cfg=dict(
                     assigner=dict(
                        type='LaneHungarianAssigner3D',
                        cls_cost=dict(type='FocalLossCost', weight=1.5),
                        reg_cost=dict(type='LaneL1Cost', weight=0.025))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 use_path_ind=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'

            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_lanes_one2one = num_lanes_one2one
        self.lanes_group = lanes_group
        self.loss_weight_group = loss_weight_group
        self.use_group_embed = use_group_embed
        self.num_point = num_point
        self.query_embed_type = query_embed_type
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.use_path_ind = use_path_ind

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.reg_embed_dims = reg_embed_dims
        self.detach_te_feat = detach_te_feat

        self.topo_head = topo_head
        if lclc_head is not None:
            self.lclc_cfg = lclc_head

        if lcte_head is not None:
            self.lcte_cfg = lcte_head

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        assert pts_dim in (2, 3)
        self.pts_dim = pts_dim

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = pts_dim * 11
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, ] * self.code_size
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.gt_c_save = self.code_size

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_reg_fcs = num_reg_fcs
        self._init_layers()

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        if self.reg_embed_dims:
            reg_branch.append(Linear(self.embed_dims, self.reg_embed_dims))
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.reg_embed_dims, self.reg_embed_dims))
                reg_branch.append(nn.ReLU())
            if self.query_embed_type == 'instance_pts':
                reg_branch.append(Linear(self.reg_embed_dims, self.code_size))
            elif self.query_embed_type == 'all_pts':
                reg_branch.append(Linear(self.reg_embed_dims, self.pts_dim))
        else:
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            if self.query_embed_type == 'instance_pts':
                reg_branch.append(Linear(self.embed_dims, self.code_size))
            elif self.query_embed_type == 'all_pts':
                reg_branch.append(Linear(self.embed_dims, self.pts_dim))
        reg_branch = nn.Sequential(*reg_branch)

        lclc_branch = build_head(self.lclc_cfg)
        lcte_branch = build_head(self.lcte_cfg)

        te_embed_branch = []
        in_channels = self.embed_dims
        for _ in range(self.num_reg_fcs - 1):
            te_embed_branch.append(nn.Sequential(
                    Linear(in_channels, 2 * self.embed_dims),
                    nn.ReLU(),
                    nn.Dropout(0.1)))
            in_channels = 2 * self.embed_dims
        te_embed_branch.append(Linear(2 * self.embed_dims, self.embed_dims))
        te_embed_branch = nn.Sequential(*te_embed_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.transformer.decoder.num_layers
        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)
        self.lclc_branches = _get_clones(lclc_branch, num_pred)
        self.lcte_branches = _get_clones(lcte_branch, num_pred)
        self.te_embed_branches = _get_clones(te_embed_branch, num_pred)

        if self.query_embed_type == 'instance_pts':
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        elif self.query_embed_type == 'all_pts':
            self.query_embedding = None
            self.instance_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            self.pts_embedding = nn.Embedding(self.num_point, self.embed_dims * 2)
        else:
            raise NotImplementedError
        if self.use_group_embed:
            self.group_embed = nn.Parameter(torch.Tensor(2, 1, self.embed_dims))

    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.use_group_embed:
            normal_(self.group_embed)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, bev_feats, img_metas, te_feats, te_cls_scores):

        dtype = mlvl_feats[0].dtype
        # device = mlvl_feats[0].device
        if self.query_embed_type == 'instance_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'all_pts':
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)

        if self.detach_te_feat:
            te_feats = te_feats.clone().detach()
        te_feats = torch.stack([self.te_embed_branches[lid](te_feats[lid]) for lid in range(len(te_feats))])

        # breakpoint()
        lane_object_query_embeds = object_query_embeds[ : self.lanes_group[0] * self.num_lanes_one2one]
        path_object_query_embeds = object_query_embeds[self.lanes_group[0] * self.num_lanes_one2one : ]
        if self.use_group_embed:
            lane_object_query_embeds = torch.cat((lane_object_query_embeds[:, :self.embed_dims] + self.group_embed[0], 
                                            lane_object_query_embeds[:, self.embed_dims:]), dim=1)
            path_object_query_embeds = torch.cat((path_object_query_embeds[:, :self.embed_dims] + self.group_embed[1], 
                                            path_object_query_embeds[:, self.embed_dims:]), dim=1)
        
        if not self.training:
            lane_object_query_embeds = lane_object_query_embeds[ : self.num_lanes_one2one]
            if self.lanes_group[1] > 0:
                path_object_query_embeds = path_object_query_embeds[ : self.num_lanes_one2one]
        
        object_query_embeds = torch.cat((lane_object_query_embeds, path_object_query_embeds), dim=0)

        outputs = self.transformer(
            mlvl_feats,
            bev_feats,
            object_query_embeds,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            cls_branches=self.cls_branches,
            reg_branches=self.reg_branches,
            lclc_branches=self.lclc_branches,
            lcte_branches=self.lcte_branches,
            te_feats=te_feats,
            te_cls_scores=te_cls_scores,
            img_metas=img_metas,
        )

        hs, init_reference, inter_references, inter_cls, inter_reg, lclc_rel_out, lcte_rel_out = outputs
        hs = hs.permute(0, 2, 1, 3)

        outs = {
            'all_cls_scores': inter_cls, # outputs_classes
            'all_lanes_preds': inter_reg, # outputs_coords
            'all_lclc_preds': lclc_rel_out,
            'all_lcte_preds': lcte_rel_out,
            'history_states': hs
        }

        return outs

    def _get_target_single(self,
                           cls_score,
                           lanes_pred,
                           lclc_pred,
                           gt_labels,
                           gt_lanes,
                           gt_lane_adj,
                           gt_bboxes_ignore=None):

        # breakpoint()
        num_bboxes = lanes_pred.size(0)
        
        # assigner and sampler
        assign_result = self.assigner.assign(lanes_pred, cls_score, gt_lanes, gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, lanes_pred, gt_lanes)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lanes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_lanes.new_ones(num_bboxes)

        # bbox targets
        gt_c = gt_lanes.shape[-1]
        if gt_c == 0:
            gt_c = self.gt_c_save
            sampling_result.pos_gt_bboxes = torch.zeros((0, gt_c)).to(sampling_result.pos_gt_bboxes.device)
        else:
            self.gt_c_save = gt_c

        bbox_targets = torch.zeros_like(lanes_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(lanes_pred)
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        bbox_weights[pos_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, pos_assigned_gt_inds)

    def get_targets(self,
                    cls_scores_list,
                    lanes_preds_list,
                    lclc_preds_list,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_lane_adj_list,
                    gt_bboxes_ignore_list=None):

        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, lanes_targets_list, lanes_weights_list,
            pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, lanes_preds_list, lclc_preds_list,
            gt_labels_list, gt_lanes_list, gt_lane_adj_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        assign_result = dict(
            pos_inds=pos_inds_list, neg_inds=neg_inds_list, pos_assigned_gt_inds=pos_assigned_gt_inds_list
        )
        return (labels_list, label_weights_list, lanes_targets_list, lanes_weights_list, num_total_pos, num_total_neg, assign_result)

    def loss_single_path(self,
                    history_states,
                    cls_scores,
                    lanes_preds,
                    lclc_preds,
                    lcte_preds,
                    te_assign_result,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_lane_adj_list,
                    gt_lane_lcte_adj_list,
                    layer_index,
                    path_ind=None,
                    gt_bboxes_ignore_list=None):
        
        # breakpoint()
        num_imgs = history_states.size(0)
        history_states_list = [history_states[i] for i in range(num_imgs)]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        lanes_preds_list = [lanes_preds[i] for i in range(num_imgs)]
        lclc_preds_list = [lclc_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, lanes_preds_list, lclc_preds_list, 
                                           gt_lanes_list, gt_labels_list, gt_lane_adj_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, assign_result) = cls_reg_targets
        pos_assigned_gt_inds = assign_result['pos_assigned_gt_inds'][0]
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        lanes_preds = lanes_preds.reshape(-1, lanes_preds.size(-1))
        isnotnan = torch.isfinite(bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            lanes_preds[isnotnan, :self.code_size], 
            bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)

        loss_rank = torch.zeros_like(loss_bbox)
        # lclc loss
        if path_ind is None:
            loss_lclc = self.lclc_branches[layer_index].loss(history_states_list, lclc_preds, lanes_preds_list, gt_lane_adj_list, gt_lanes_list, assign_result, assign_result)
        else:
            loss_lclc = self.lclc_branches[layer_index].loss(history_states_list, lclc_preds, lanes_preds_list, gt_lane_adj_list, gt_lanes_list, assign_result, assign_result, path_ind)
        if 'loss_rank' in loss_lclc.keys():
            loss_rank += loss_lclc['loss_rank']
        if 'loss_ll_l1' in loss_lclc.keys():
            loss_lclc = loss_lclc['loss_ll_l1'] + loss_lclc['loss_rel']
        else:
            loss_lclc = loss_lclc['loss_rel']

        loss_lcte = self.lcte_branches[layer_index].loss(lcte_preds, gt_lane_lcte_adj_list, assign_result, te_assign_result)
        if 'loss_rank' in loss_lcte.keys():
            loss_rank += loss_lcte['loss_rank']
        loss_lcte = loss_lcte['loss_rel']

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, loss_lclc, loss_lcte, loss_rank, [labels, pos_assigned_gt_inds]

    def loss_single(self,
                    history_states,
                    cls_scores,
                    lanes_preds,
                    lclc_preds,
                    lcte_preds,
                    te_assign_result,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_lane_adj_list,
                    gt_lane_lcte_adj_list,
                    layer_index,
                    gt_bboxes_ignore_list=None):
        
        # breakpoint()
        num_imgs = history_states.size(0)
        history_states_list = [history_states[i] for i in range(num_imgs)]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        lanes_preds_list = [lanes_preds[i] for i in range(num_imgs)]
        lclc_preds_list = [lclc_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, lanes_preds_list, lclc_preds_list, 
                                           gt_lanes_list, gt_labels_list, gt_lane_adj_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, assign_result) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        lanes_preds = lanes_preds.reshape(-1, lanes_preds.size(-1))
        isnotnan = torch.isfinite(bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            lanes_preds[isnotnan, :self.code_size], 
            bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)

        loss_rank = torch.zeros_like(loss_bbox)
        # lclc loss
        loss_lclc = self.lclc_branches[layer_index].loss(history_states_list, lclc_preds, lanes_preds_list, gt_lane_adj_list, gt_lanes_list, assign_result, assign_result)
        if 'loss_rank' in loss_lclc.keys():
            loss_rank += loss_lclc['loss_rank']
        if 'loss_ll_l1' in loss_lclc.keys():
            loss_lclc = loss_lclc['loss_ll_l1'] + loss_lclc['loss_rel']
        else:
            loss_lclc = loss_lclc['loss_rel']

        loss_lcte = self.lcte_branches[layer_index].loss(lcte_preds, gt_lane_lcte_adj_list, assign_result, te_assign_result)
        if 'loss_rank' in loss_lcte.keys():
            loss_rank += loss_lcte['loss_rank']
        loss_lcte = loss_lcte['loss_rel']

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, loss_lclc, loss_lcte, loss_rank

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             gt_lanes_3d, # [(N, 33), ]
             gt_labels_list, # [(N), ]
             gt_lane_adj, # [(N, N), ]
             gt_lane_lcte_adj, # [(N, M), ]
             te_assign_results,
             gt_bboxes_ignore=None,
             img_metas=None):

        # breakpoint()
        all_history_states = preds_dicts['history_states'] # 6, b, l * k, 256
        all_cls_scores = preds_dicts['all_cls_scores'] # 6, b, l * k, 1
        all_lanes_preds = preds_dicts['all_lanes_preds'] # 6, b, l * k, 33
        all_lclc_preds = preds_dicts['all_lclc_preds'] # 6, b, k, l, l, 1
        all_lcte_preds = preds_dicts['all_lcte_preds'] # 6, b, k, l, t, 1

        num_dec_layers, bs = all_lclc_preds.size(0), all_lclc_preds.size(1)
        layer_index = [i for i in range(num_dec_layers)]

        # one
        all_gt_lanes_list = [gt_lanes_3d for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_lane_adj_list = [gt_lane_adj for _ in range(num_dec_layers)]
        all_gt_lane_lcte_adj_list = [gt_lane_lcte_adj for _ in range(num_dec_layers)]

        # many
        one2many_gt_lanes_3d = []
        one2many_gt_labels_list = []
        one2many_gt_lane_adj = []
        one2many_gt_lane_lcte_adj = []
        
        for (_gt_lanes_3d, _gt_labels_list, _gt_lane_adj, _gt_lane_lcte_adj) in zip(gt_lanes_3d, gt_labels_list, gt_lane_adj, gt_lane_lcte_adj):
            row_idx, col_idx = torch.where(_gt_lane_adj == 1)
            row_col_pair = torch.stack([row_idx, col_idx], dim=-1)
            if len(row_idx):
                prev_lane, next_lane = _gt_lanes_3d[row_idx], _gt_lanes_3d[col_idx]
                new_gt_lanes_3d = torch.cat((prev_lane, next_lane[:, self.pts_dim:]), dim=1).view(len(row_idx), -1, self.pts_dim)
                new_gt_lanes_3d = new_gt_lanes_3d[:, ::2, :].contiguous().view(len(row_idx), -1)
            else:
                new_gt_lanes_3d = _gt_lanes_3d
                row_idx = torch.arange(len(_gt_lanes_3d), dtype=torch.int64).to(_gt_lanes_3d.device)
                col_idx = torch.arange(len(_gt_lanes_3d), dtype=torch.int64).to(_gt_lanes_3d.device)

            new_gt_labels_list = _gt_labels_list.new_zeros((len(new_gt_lanes_3d), ))
            new_gt_lane_adj = _gt_lane_adj[col_idx][:, row_idx] # next lane -> prev lane
            new_gt_lane_lcte_adj = _gt_lane_lcte_adj[col_idx] # next lane -> traffic element
            
            one2many_gt_lanes_3d.append(new_gt_lanes_3d)
            one2many_gt_labels_list.append(new_gt_labels_list)
            one2many_gt_lane_adj.append(new_gt_lane_adj)
            one2many_gt_lane_lcte_adj.append(new_gt_lane_lcte_adj)

        all_gt_lanes_list_one2many = [one2many_gt_lanes_3d for _ in range(num_dec_layers)]
        all_gt_labels_list_one2many = [one2many_gt_labels_list for _ in range(num_dec_layers)]
        all_gt_lane_adj_list_one2many = [one2many_gt_lane_adj for _ in range(num_dec_layers)]
        all_gt_lane_lcte_adj_list_one2many = [one2many_gt_lane_lcte_adj for _ in range(num_dec_layers)]
        
        if self.use_path_ind:
            # just for one2many_labels
            for group_idx in range(self.lanes_group[0], self.lanes_group[0] + self.lanes_group[1]):
                losses_cls_one2many, losses_bbox_one2many, losses_lclc_one2many, losses_lcte_one2many, _, one2many_labels = multi_apply(self.loss_single_path, 
                    all_history_states[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
                    all_cls_scores[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
                    all_lanes_preds[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
                    all_lclc_preds[:, :, group_idx, :, :, :], 
                    all_lcte_preds[:, :, group_idx, :, :, :], 
                    te_assign_results,
                    all_gt_lanes_list_one2many, all_gt_labels_list_one2many, all_gt_lane_adj_list_one2many, all_gt_lane_lcte_adj_list_one2many, layer_index)
            
            tmp_path_preds = all_lanes_preds[:, :, self.num_lanes_one2one * self.lanes_group[0] : self.num_lanes_one2one * (self.lanes_group[0] + 1), :]
            tmp_lane_preds = all_lanes_preds[:, :, self.num_lanes_one2one * 0 : self.num_lanes_one2one * (0 + 1), :].detach()
            last_tmp_path_preds = tmp_path_preds[-1][0]
            last_tmp_lane_preds = tmp_lane_preds[-1][0]

            one2many_labels, pos_assigned_gt_inds = one2many_labels[-1]
            if len(row_col_pair):
                row_col_pair = row_col_pair[pos_assigned_gt_inds]

            last_one2many_labels = one2many_labels
            select_tmp_path_preds = last_tmp_path_preds[last_one2many_labels == 0].detach()
            select_tmp_path_preds = select_tmp_path_preds.view(len(select_tmp_path_preds), -1, 3)
            o1_reg = select_tmp_path_preds[:, :6].view(len(select_tmp_path_preds), -1) # 0, 1, 2, 3, 4, 5
            o2_reg = select_tmp_path_preds[:, 5:].view(len(select_tmp_path_preds), -1) # 5, 6, 7, 8, 9, 10

            _inter_reg = last_tmp_lane_preds.view(len(last_tmp_lane_preds), -1, 3)
            _inter_reg = _inter_reg[:, ::2].contiguous().view(len(_inter_reg), -1) # 0, 2, 4, 6, 8, 10

            cost_1 = torch.cdist(_inter_reg, o1_reg, p=1)
            cost_2 = torch.cdist(_inter_reg, o2_reg, p=1)
            _,ind_1 = torch.min(cost_1, dim=0)
            _,ind_2 = torch.min(cost_2, dim=0)
            select_pos_ind = torch.unique(torch.cat([ind_1, ind_2],dim=0))

            # get new topo gt
            new_topo_gt = torch.zeros([row_col_pair.shape[0] * 2, row_col_pair.shape[0] * 2], dtype=row_col_pair.dtype, device=row_col_pair.device)
            
            # ind2ind
            ind2ind = {}
            for _ind, row_col in enumerate(row_col_pair):
                if int(row_col[0]) not in ind2ind:
                    ind2ind[int(row_col[0])] = []
                ind2ind[int(row_col[0])].append(_ind)
                if int(row_col[1]) not in ind2ind:
                    ind2ind[int(row_col[1])] = []
                ind2ind[int(row_col[1])].append(_ind + len(row_col_pair))

            for _ind, row_col in enumerate(row_col_pair):
                _from_ind, _to_ind = row_col
                _from_ind, _to_ind = int(_from_ind), int(_to_ind)
                froms_inds = ind2ind[_from_ind]
                to_inds = ind2ind[_to_ind]
                for _x in froms_inds:
                    for _y in to_inds:
                        new_topo_gt[_x][_y] = 1

            new_pack = (torch.cat([ind_1, ind_2]), new_topo_gt)
            select_pos_ind = (select_pos_ind,new_pack)
        else:
            select_pos_ind = None

        # loss
        group_idx = 0
        losses_cls, losses_bbox, losses_lclc, losses_lcte, losses_rank, _ = multi_apply(self.loss_single_path, 
            all_history_states[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
            all_cls_scores[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
            all_lanes_preds[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
            all_lclc_preds[:, :, group_idx, :, :, :], 
            all_lcte_preds[:, :, group_idx, :, :, :], 
            te_assign_results,
            all_gt_lanes_list, all_gt_labels_list, all_gt_lane_adj_list, all_gt_lane_lcte_adj_list, layer_index, [select_pos_ind] * len(layer_index))

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict['loss_lane_cls'] = losses_cls[-1]
        loss_dict['loss_lane_reg'] = losses_bbox[-1]
        loss_dict['loss_lclc_rel'] = losses_lclc[-1]
        loss_dict['loss_lcte_rel'] = losses_lcte[-1]
        loss_dict['loss_rank_rel'] = losses_rank[-1]
        if self.lanes_group[0] + self.lanes_group[1] > 1:
            loss_dict['loss_lane_cls_h'] = 0.
            loss_dict['loss_lane_reg_h'] = 0.
            loss_dict['loss_lclc_rel_h'] = 0.
            loss_dict['loss_lcte_rel_h'] = 0.
            loss_dict['loss_rank_rel_h'] = 0.

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_lclc_i, loss_lcte_i, loss_rank_i in zip(
                losses_cls[:-1], losses_bbox[:-1], losses_lclc[:-1], losses_lcte[:-1], losses_rank[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_lane_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_lane_reg'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_lclc_rel'] = loss_lclc_i
            loss_dict[f'd{num_dec_layer}.loss_lcte_rel'] = loss_lcte_i
            loss_dict[f'd{num_dec_layer}.loss_rank_rel'] = loss_rank_i
            if self.lanes_group[0] + self.lanes_group[1] > 1:
                loss_dict[f'd{num_dec_layer}.loss_lane_cls_h'] = 0.
                loss_dict[f'd{num_dec_layer}.loss_lane_reg_h'] = 0.
                loss_dict[f'd{num_dec_layer}.loss_lclc_rel_h'] = 0.
                loss_dict[f'd{num_dec_layer}.loss_lcte_rel_h'] = 0.
                loss_dict[f'd{num_dec_layer}.loss_rank_rel_h'] = 0.
            num_dec_layer += 1

        for group_idx in range(1, self.lanes_group[0]):
            losses_cls, losses_bbox, losses_lclc, losses_lcte, losses_rank = multi_apply(self.loss_single, 
                all_history_states[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
                all_cls_scores[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
                all_lanes_preds[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
                all_lclc_preds[:, :, group_idx, :, :, :], 
                all_lcte_preds[:, :, group_idx, :, :, :], 
                te_assign_results,
                all_gt_lanes_list, all_gt_labels_list, all_gt_lane_adj_list, all_gt_lane_lcte_adj_list, layer_index)

            # loss from the last decoder layer
            loss_dict['loss_lane_cls_h'] += losses_cls[-1] * self.loss_weight_group
            loss_dict['loss_lane_reg_h'] += losses_bbox[-1] * self.loss_weight_group
            loss_dict['loss_lclc_rel_h'] += losses_lclc[-1] * self.loss_weight_group
            loss_dict['loss_lcte_rel_h'] += losses_lcte[-1] * self.loss_weight_group
            loss_dict['loss_rank_rel_h'] += losses_rank[-1] * self.loss_weight_group

            # loss from other decoder layers            
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_lclc_i, loss_lcte_i, loss_rank_i in zip(
                    losses_cls[:-1], losses_bbox[:-1], losses_lclc[:-1], losses_lcte[:-1], losses_rank[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_lane_cls_h'] += loss_cls_i * self.loss_weight_group
                loss_dict[f'd{num_dec_layer}.loss_lane_reg_h'] += loss_bbox_i * self.loss_weight_group
                loss_dict[f'd{num_dec_layer}.loss_lclc_rel_h'] += loss_lclc_i * self.loss_weight_group
                loss_dict[f'd{num_dec_layer}.loss_lcte_rel_h'] += loss_lcte_i * self.loss_weight_group
                loss_dict[f'd{num_dec_layer}.loss_rank_rel_h'] += loss_rank_i * self.loss_weight_group
                num_dec_layer += 1

        for group_idx in range(self.lanes_group[0], self.lanes_group[0] + self.lanes_group[1]):
            losses_cls_one2many, losses_bbox_one2many, losses_lclc_one2many, losses_lcte_one2many, _ = multi_apply(self.loss_single, 
                all_history_states[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
                all_cls_scores[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
                all_lanes_preds[:, :, self.num_lanes_one2one * group_idx : self.num_lanes_one2one * (group_idx + 1), :], 
                all_lclc_preds[:, :, group_idx, :, :, :], 
                all_lcte_preds[:, :, group_idx, :, :, :], 
                te_assign_results,
                all_gt_lanes_list_one2many, all_gt_labels_list_one2many, all_gt_lane_adj_list_one2many, all_gt_lane_lcte_adj_list_one2many, layer_index)
            
            # loss from the last decoder layer
            loss_dict['loss_lane_cls_h'] += losses_cls_one2many[-1] * self.loss_weight_group
            loss_dict['loss_lane_reg_h'] += losses_bbox_one2many[-1] * self.loss_weight_group
            loss_dict['loss_lclc_rel_h'] += losses_lclc_one2many[-1] * self.loss_weight_group * 0
            loss_dict['loss_lcte_rel_h'] += losses_lcte_one2many[-1] * self.loss_weight_group * 0

            # loss from other decoder layers            
            num_dec_layer = 0
            for loss_cls_one2many_i, loss_bbox_one2many_i, loss_lclc_one2many_i, loss_lcte_one2many_i in zip(
                    losses_cls_one2many[:-1], losses_bbox_one2many[:-1], losses_lclc_one2many[:-1], losses_lcte_one2many[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_lane_cls_h'] += loss_cls_one2many_i * self.loss_weight_group
                loss_dict[f'd{num_dec_layer}.loss_lane_reg_h'] += loss_bbox_one2many_i * self.loss_weight_group
                loss_dict[f'd{num_dec_layer}.loss_lclc_rel_h'] += loss_lclc_one2many_i * self.loss_weight_group * 0
                loss_dict[f'd{num_dec_layer}.loss_lcte_rel_h'] += loss_lcte_one2many_i * self.loss_weight_group * 0
                num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, rescale=False):

        # all_lclc_preds = preds_dicts['all_lclc_preds'][-1].squeeze(-1).sigmoid().detach().cpu().numpy()
        all_lclc_preds = preds_dicts['all_lclc_preds'][-1].detach().cpu().numpy()
        all_lclc_preds = [_ for _ in all_lclc_preds]

        # all_lcte_preds = preds_dicts['all_lcte_preds'][-1].squeeze(-1).sigmoid().detach().cpu().numpy()
        all_lcte_preds = preds_dicts['all_lcte_preds'][-1].detach().cpu().numpy()
        all_lcte_preds = [_ for _ in all_lcte_preds]

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            lanes = preds['lane3d']
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([lanes, scores, labels])
        return ret_list, all_lclc_preds, all_lcte_preds
