import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_transformer_layer
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid


@HEADS.register_module()
class BEVSegHead(nn.Module):
    def __init__(self,
                 bev_h=100,
                 bev_w=200,
                 embed_dims=256,
                 seg_classes=1,
                 pos_weight=4.0,
                 loss_weight=1.0):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.seg_classes = seg_classes
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, self.seg_classes, kernel_size=1, padding=0)
        )
        
        self.loss_seg = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.loss_weight = loss_weight

    def forward(self, bev_embed):
        bs = bev_embed.size(0) # bs, h * w, 256
        seg_bev_embed = bev_embed.view(bs, self.bev_h, self.bev_w, self.embed_dims)
        seg_bev_embed = seg_bev_embed.permute(0, 3, 1, 2).contiguous() # bs, 256, h, w
        outputs_seg = self.seg_head(seg_bev_embed)
        return outputs_seg
    
    def loss(self, outputs_seg, gt_seg_mask):
        bs = outputs_seg.size(0)
        seg_gt = torch.stack([gt_seg_mask[i] for i in range(bs)], dim=0)
        loss_seg = self.loss_seg(outputs_seg, seg_gt.float())
        return loss_seg * self.loss_weight