import copy
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mmcv
from mmcv.cnn import Linear, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout 
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, FEEDFORWARD_NETWORK,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn import xavier_init
from scipy.optimize import linear_sum_assignment
from .position_embed import gen_sineembed_for_position
from ...core.lane.util import fix_pts_interpolate


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class UniTopoDecoder(TransformerLayerSequence):

    def __init__(self, 
                 *args, 
                 return_intermediate=False, 
                 pc_range=None, 
                 num_points=11,
                 pts_dim=3, 
                 topo_head='toponet', 
                 num_lanes_one2one=300,
                 lanes_group=None,
                 use_attn_mask=False,
                 with_box_refine=False,
                 with_multi_point=False,
                 **kwargs):

        super(UniTopoDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.pc_range = pc_range
        self.num_points = num_points
        self.pts_dim = pts_dim
        self.topo_head = topo_head
        self.num_lanes_one2one = num_lanes_one2one
        self.lanes_group = lanes_group
        self.use_attn_mask = use_attn_mask
        if self.use_attn_mask:
            if self.use_attn_mask == 2:
                self.attn_mlp = MLP(2, 16, 8, 3)
            elif self.use_attn_mask == 3:
                import copy
                def _get_clones(module, N):
                    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
                _score_embedding = MLP(1, 16, 8, 2)
                self.attn_mlp = _get_clones(_score_embedding, 5)
            else:
                self.attn_mlp = MLP(1, 16, 8, 2)
        self.with_box_refine = with_box_refine
        self.with_multi_point = with_multi_point
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                cls_branches=None,
                reg_branches=None,
                lclc_branches=None,
                lcte_branches=None,
                key_padding_mask=None,
                te_feats=None,
                te_cls_scores=None,
                **kwargs):

        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_cls = []
        intermediate_reg = []
        intermediate_lclc_rel = []
        intermediate_lcte_rel = []
        num_query, bs = query.size(0), query.size(1)
        num_te_query = te_feats.size(2)

        # breakpoint()
        prev_lclc_adj = torch.zeros((bs, num_query // self.num_lanes_one2one, self.num_lanes_one2one, self.num_lanes_one2one),
                                  dtype=query.dtype, device=query.device)
        prev_lcte_adj = torch.zeros((bs, num_query // self.num_lanes_one2one, self.num_lanes_one2one, num_te_query),
                                  dtype=query.dtype, device=query.device)
        self_attn_masks = None
        if self.with_box_refine:
            # breakpoint()
            last_layer_coord = reference_points.clone().unsqueeze(2)
        elif self.with_multi_point:
            # breakpoint()
            if reference_points.size(-1) == self.pts_dim:
                reference_points = reference_points.unsqueeze(2).repeat(1, 1, self.num_points, 1) # identical_ref
            else:
                reference_points = reference_points.unsqueeze(2).reshape(bs, num_query, self.num_points, 3)
        for lid, layer in enumerate(self.layers):
            if self.with_multi_point:
                reference_points_input = reference_points[..., :2] # BS NUM_QUERY NUM_POINT 2
            else:
                reference_points_input = reference_points[..., :2].unsqueeze(2) # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                te_query=te_feats[lid],
                te_cls_scores=te_cls_scores[lid],
                lclc_adj=prev_lclc_adj,
                lcte_adj=prev_lcte_adj,
                attn_masks=self_attn_masks,
                **kwargs)
            output = output.permute(1, 0, 2)

            assert cls_branches is not None and reg_branches is not None
            outputs_class = cls_branches[lid](output)
            tmp = reg_branches[lid](output)

            if self.with_box_refine:
                # breakpoint()
                bs, num_query, _ = tmp.shape
                tmp = tmp.view(bs, num_query, -1, self.pts_dim)
                tmp = tmp + inverse_sigmoid(last_layer_coord)
                tmp = tmp.sigmoid()
                last_layer_coord = tmp.clone().detach()
                reference_points = tmp.clone().detach()[:, :, 5] # box refine v2
            elif self.with_multi_point:
                # breakpoint()
                reference = reference_points.clone()
                reference = inverse_sigmoid(reference)
                assert reference.shape[-1] == self.pts_dim
                
                bs, num_query, _ = tmp.shape
                tmp = tmp.view(bs, num_query, -1, self.pts_dim)
                tmp = tmp + reference
                tmp = tmp.sigmoid()
                reference_points = tmp.clone().detach()
            else:
                reference = reference_points.clone()
                reference = inverse_sigmoid(reference)
                assert reference.shape[-1] == self.pts_dim
                
                bs, num_query, _ = tmp.shape
                tmp = tmp.view(bs, num_query, -1, self.pts_dim)
                tmp = tmp + reference.unsqueeze(2)
                tmp = tmp.sigmoid()

            coord = tmp.clone()
            coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            if self.pts_dim == 3:
                coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outputs_coord = coord.view(bs, num_query, -1).contiguous()

            # breakpoint()
            output = torch.cat(output.split(self.num_lanes_one2one, dim=1), dim=0)
            outputs_class = torch.cat(outputs_class.split(self.num_lanes_one2one, dim=1), dim=0)
            outputs_coord = torch.cat(outputs_coord.split(self.num_lanes_one2one, dim=1), dim=0)

            # compute coord
            pack_ind = None
            if outputs_coord.shape[0] > 1:
                select_tmp_path_preds = outputs_coord[1].detach().view(len(outputs_coord[1]), -1, self.pts_dim)
                last_tmp_lane_preds = outputs_coord[0].view(len(outputs_coord[0]), -1, self.pts_dim)
                o1_reg = select_tmp_path_preds[:, :6].view(len(select_tmp_path_preds), -1) # 0, 1, 2, 3, 4, 5
                o2_reg = select_tmp_path_preds[:, 5:].view(len(select_tmp_path_preds), -1) # 5, 6, 7, 8, 9, 10

                _inter_reg = last_tmp_lane_preds.view(len(last_tmp_lane_preds), -1, self.pts_dim)
                _inter_reg = _inter_reg[:, ::2].contiguous().view(len(_inter_reg), -1) # 0, 2, 4, 6, 8, 10

                cost_1 = torch.cdist(_inter_reg, o1_reg, p=1)
                cost_2 = torch.cdist(_inter_reg, o2_reg, p=1)
                _,ind_1 = torch.min(cost_1, dim=0)
                _,ind_2 = torch.min(cost_2, dim=0)

                path_scores = outputs_class[1][:, 0].sigmoid().detach()
                pack_ind = (ind_1, ind_2, path_scores)

            lclc_rel_out, G_topo = lclc_branches[lid](output, output, outputs_coord, outputs_coord) # TopoLLHeadGeoDistRelLossV1
            lclc_rel_out = torch.stack(lclc_rel_out.split(bs, dim=0), dim=1) # (b, k, l, l, 1), without sigmoid
            G_topo = torch.stack(G_topo.split(bs, dim=0), dim=1) # (b, k, l, l, 1), with sigmoid

            prev_lclc_adj = G_topo.clone() # .detach()
            prev_lclc_adj = prev_lclc_adj.squeeze(-1) # .sigmoid()

            lcte_rel_out = lcte_branches[lid](output, te_feats[lid].repeat(output.size(0) // bs, 1, 1), kwargs['img_metas']) # TopoLTHead
            lcte_rel_out = torch.stack(lcte_rel_out.split(bs, dim=0), dim=1) # (b, k, l, t, 1), without sigmoid

            prev_lcte_adj = lcte_rel_out.clone().detach()
            prev_lcte_adj = prev_lcte_adj.squeeze(-1).sigmoid()

            if not self.training:
                lclc_rel_out = G_topo[:, 0].squeeze(-1) # .sigmoid()
                lcte_rel_out = lcte_rel_out[:, 0].squeeze(-1).sigmoid()

            if self.use_attn_mask:
                # breakpoint()
                if self.training:
                    topo_reg = outputs_coord[bs * self.lanes_group[0] : bs * (1 + self.lanes_group[0]), :, :] # b, n, 33
                else:
                    topo_reg = outputs_coord[bs * 1 : bs * 2, :, :] # b, n, 33
                inter_reg = outputs_coord[ : bs] # bs, n, 33
                
                self_attn_masks = []
                for batch_idx in range(bs):
                    _topo_reg = topo_reg[batch_idx].clone().detach() # n, 33
                    _inter_reg = inter_reg[batch_idx].clone().detach() # m, 33

                    _topo_reg = _topo_reg.view(len(_topo_reg), -1, self.pts_dim)
                    o1_reg = _topo_reg[:, :6].view(len(_topo_reg), -1) # 0, 1, 2, 3, 4, 5
                    o2_reg = _topo_reg[:, 5:].view(len(_topo_reg), -1) # 5, 6, 7, 8, 9, 10
                    
                    _inter_reg = _inter_reg.view(len(_inter_reg), -1, self.pts_dim)
                    _inter_reg = _inter_reg[:, ::2].contiguous().view(len(_inter_reg), -1) # 0, 2, 4, 6, 8, 10

                    cost_1 = torch.cdist(_inter_reg, o1_reg, p=1)
                    cost_2 = torch.cdist(_inter_reg, o2_reg, p=1)

                    if self.use_attn_mask == 2:
                        cost = torch.stack((cost_1, cost_2), dim=-1)
                    else:
                        cost, _ = torch.min(torch.stack((cost_1, cost_2), dim=-1), dim=-1, keepdim=True)

                    if self.use_attn_mask == 3:
                        if lid == 5:
                            # last layer
                            pass
                        else:
                            attn_mask = self.attn_mlp[lid](cost)
                    else:
                        attn_mask = self.attn_mlp(cost)

                    attn_mask = attn_mask.sigmoid()
                    attn_mask = torch.clamp(attn_mask, min=1e-5)
                    attn_mask = torch.log(attn_mask) # softmax
                    attn_mask = attn_mask.permute(2, 0, 1) # head, n, n
                    self_attn_masks.append(attn_mask)
                
                self_attn_masks = [None, torch.cat(self_attn_masks, dim=0), None]

            output = torch.cat(output.split(bs, dim=0), dim=1)
            outputs_class = torch.cat(outputs_class.split(bs, dim=0), dim=1)
            outputs_coord = torch.cat(outputs_coord.split(bs, dim=0), dim=1)
            if not self.training:
                outputs_class = outputs_class[:, :self.num_lanes_one2one]
                outputs_coord = outputs_coord[:, :self.num_lanes_one2one]
                # outputs_class = outputs_class[:, self.num_lanes_one2one:]
                # outputs_coord = outputs_coord[:, self.num_lanes_one2one:]

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_cls.append(outputs_class)
                intermediate_reg.append(outputs_coord)
                intermediate_lclc_rel.append(lclc_rel_out)
                intermediate_lcte_rel.append(lcte_rel_out)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), torch.stack(
                intermediate_cls), torch.stack(
                intermediate_reg), torch.stack(
                intermediate_lclc_rel), torch.stack(
                intermediate_lcte_rel)

        return output, reference_points, outputs_class, outputs_coord, lclc_rel_out, lcte_rel_out


@TRANSFORMER_LAYER.register_module()
class UniTopoDecoderLayer(BaseTransformerLayer):

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 num_lanes_one2one=300,
                 lanes_group=None,
                 **kwargs):

        super(UniTopoDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            **kwargs)
        # assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.num_lanes_one2one = num_lanes_one2one
        self.lanes_group = lanes_group
    
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                te_query=None,
                te_cls_scores=None,
                lclc_adj=None,
                lcte_adj=None,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        # breakpoint()
        for layer_idx, layer in enumerate(self.operation_order):
            if layer == 'self_attn' and attn_index == 0:
                bs = query.size(1)
                if len(query) > self.num_lanes_one2one:
                    query = torch.cat(query.split(self.num_lanes_one2one, dim=0), dim=1)
                    identity = torch.cat(identity.split(self.num_lanes_one2one, dim=0), dim=1)
                    query_pos = torch.cat(query_pos.split(self.num_lanes_one2one, dim=0), dim=1)
                    
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    
                    query = torch.cat(query.split(bs, dim=1), dim=0)
                    identity = torch.cat(identity.split(bs, dim=1), dim=0)
                    query_pos = torch.cat(query_pos.split(bs, dim=1), dim=0)
                else:
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'self_attn' and attn_index == 1:
                # breakpoint()
                bs = query.size(1)
                query = list(query.split(self.num_lanes_one2one, dim=0))
                query_pos = list(query_pos.split(self.num_lanes_one2one, dim=0))
                if self.training:
                    temp_query = self.attentions[attn_index](
                        query[0], # query
                        query[self.lanes_group[0]], # key
                        query[self.lanes_group[0]], # value
                        identity=None,
                        query_pos=query_pos[0], # query
                        key_pos=query_pos[self.lanes_group[0]], # key
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                else:
                    temp_query = self.attentions[attn_index](
                        query[0], # query
                        query[1], # key
                        query[1], # value
                        identity=None,
                        query_pos=query_pos[0], # query
                        key_pos=query_pos[1], # key
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)

                query[0] = temp_query
                query = torch.cat(query, dim=0)
                query_pos = torch.cat(query_pos, dim=0)
                
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, te_query, lclc_adj, lcte_adj, te_cls_scores, identity=identity if self.pre_norm else None)
                ffn_index += 1

        return query

@FEEDFORWARD_NETWORK.register_module()
class UniTopo_FFN(BaseModule):

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=512,
                 num_query=200,
                 num_point=11,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.1,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 edge_weight=0.5, 
                 num_te_classes=13,
                 num_lanes_one2one=300,
                 many_gcn=False,
                 detach_te_feat=False,
                 **kwargs):

        super(UniTopo_FFN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_query = num_query
        self.num_point = num_point
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(
            Sequential(
                Linear(feedforward_channels, embed_dims), self.activate,
                nn.Dropout(ffn_drop)))
        self.layers = Sequential(*layers)
        self.num_lanes_one2one = num_lanes_one2one
        self.many_gcn = many_gcn
        self.edge_weight = edge_weight

        self.lclc_gnn_layer = LclcSkgGCNLayer(embed_dims, embed_dims, edge_weight=edge_weight)
        self.lcte_gnn_layer = LcteSkgGCNLayer(embed_dims, embed_dims, 
                                num_te_classes=num_te_classes, edge_weight=edge_weight, detach_te_feat=detach_te_feat)

        self.downsample = nn.Linear(embed_dims * 2, embed_dims)

        self.gnn_dropout1 = nn.Dropout(ffn_drop)
        self.gnn_dropout2 = nn.Dropout(ffn_drop)

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, lc_query, te_query, lclc_adj, lcte_adj, te_cls_scores, identity=None):
        # breakpoint()
        out = self.layers(lc_query)
        out = out.permute(1, 0, 2)

        '''
            out: torch.Size([b, 300 * k, 256])
            te_query: torch.Size([b, 100, 256])
            te_cls_scores: torch.Size([b, 100, 13])

            lclc_adj: torch.Size([b, k, 300, 300])
            lcte_adj: torch.Size([b, k, 300, 100])
        '''
        bs = out.size(0)
        out = out.split(self.num_lanes_one2one, dim=1) # [b, 300, 256] * k
        
        out_one = out[0] # b, 300, 256
        lclc_adj_one = lclc_adj[:, 0] # b, 300, 300
        lcte_adj_one = lcte_adj[:, 0] # b, 300, 100
        lclc_features_one = self.lclc_gnn_layer(out_one, lclc_adj_one)
        lcte_features_one = self.lcte_gnn_layer(te_query, lcte_adj_one, te_cls_scores)
        out_one = torch.cat([lclc_features_one, lcte_features_one], dim=-1)
        out_one = self.activate(out_one)
        out_one = self.gnn_dropout1(out_one)
        out_one = self.downsample(out_one)
        out_one = self.gnn_dropout2(out_one)

        if len(out) > 1:
            out_many = torch.cat(out[1:], dim=0) # b * (k - 1), 300, 256
            lclc_adj_many = lclc_adj[:, 1:].flatten(0, 1) # b * (k - 1), 300, 300
            lcte_adj_many = lcte_adj[:, 1:].flatten(0, 1) # b * (k - 1), 300, 100
            if self.many_gcn:
                lclc_features_many = self.lclc_gnn_layer(out_many, lclc_adj_many)
                lcte_features_many = self.lcte_gnn_layer(te_query.repeat(lcte_adj_many.size(0) // bs, 1, 1), 
                                                         lcte_adj_many, 
                                                         te_cls_scores.repeat(lcte_adj_many.size(0) // bs, 1, 1))
                out_many = torch.cat([lclc_features_many, lcte_features_many], dim=-1)
                out_many = self.activate(out_many)
                out_many = self.gnn_dropout1(out_many)
                out_many = self.downsample(out_many)
                out_many = self.gnn_dropout2(out_many)

            out = torch.cat([out_one, out_many], dim=0)
        else:
            out = out_one

        out = torch.cat(out.split(bs, dim=0), dim=1)
        out = out.permute(1, 0, 2)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = lc_query
        return identity + self.dropout_layer(out)


class LclcSkgGCNLayer(nn.Module):

    def __init__(self, in_features, out_features, edge_weight=0.5):
        super(LclcSkgGCNLayer, self).__init__()
        self.edge_weight = edge_weight

        if self.edge_weight != 0:
            self.weight_forward = torch.Tensor(in_features, out_features)
            self.weight_forward = nn.Parameter(nn.init.xavier_uniform_(self.weight_forward))
            self.weight_backward = torch.Tensor(in_features, out_features)
            self.weight_backward = nn.Parameter(nn.init.xavier_uniform_(self.weight_backward))

        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.edge_weight = edge_weight

    def forward(self, input, adj):

        support_loop = torch.matmul(input, self.weight)
        output = support_loop

        if self.edge_weight != 0:
            support_forward = torch.matmul(input, self.weight_forward)
            output_forward = torch.matmul(adj, support_forward)
            output += self.edge_weight * output_forward

            support_backward = torch.matmul(input, self.weight_backward)
            output_backward = torch.matmul(adj.permute(0, 2, 1), support_backward)
            output += self.edge_weight * output_backward

        return output

class LclcSkgGCNLayerOneway(nn.Module):

    def __init__(self, in_features, out_features, edge_weight=0.5):
        super(LclcSkgGCNLayerOneway, self).__init__()    
        self.weight_forward = torch.Tensor(in_features, out_features)
        self.weight_forward = nn.Parameter(nn.init.xavier_uniform_(self.weight_forward))
        self.edge_weight = edge_weight

    def forward(self, input, adj):
        support_forward = torch.matmul(input, self.weight_forward)
        output_forward = torch.matmul(adj, support_forward)
        output = self.edge_weight * output_forward
        return output

class LcteSkgGCNLayer(nn.Module):

    def __init__(self, in_features, out_features, num_te_classes=13, edge_weight=0.5, detach_te_feat=False):
        super(LcteSkgGCNLayer, self).__init__()
        self.weight = torch.Tensor(num_te_classes, in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.edge_weight = edge_weight
        self.detach_te_feat = detach_te_feat

    def forward(self, te_query, lcte_adj, te_cls_scores):
        # te_cls_scores: (bs, num_te_query, num_te_classes)
        cls_scores = te_cls_scores.detach().sigmoid().unsqueeze(3)
        # te_query: (bs, num_te_query, embed_dims)
        # (bs, num_te_query, 1, embed_dims) * (bs, num_te_query, num_te_classes, 1)
        te_feats = te_query.unsqueeze(2) * cls_scores
        if self.detach_te_feat:
            te_feats = te_feats.clone().detach()
        # (bs, num_te_classes, num_te_query, embed_dims)
        te_feats = te_feats.permute(0, 2, 1, 3)

        support = torch.matmul(te_feats, self.weight).sum(1)
        adj = lcte_adj * self.edge_weight
        output = torch.matmul(adj, support)
        return output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx):
        # breakpoint()
        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), input.size(1), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        x_reset, x_upd, x_new = x_t.chunk(3, 2)
        h_reset, h_upd, h_new = h_t.chunk(3, 2)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy