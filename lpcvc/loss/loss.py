import torch
import math
import torch.nn as nn
import numpy as np
import scipy.ndimage as nd

from typing import Optional
from torch.nn import functional as F
from torch.autograd import Variable

__all__ = ['CrossEntropyLoss', 'OhemCELoss2D']


# model train loss
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self,
                 weight=None,
                 ignore_index=-1):

        super(CrossEntropyLoss, self).__init__(weight, None, ignore_index)

    def forward(self, pred, target):
            return super(CrossEntropyLoss, self).forward(pred, target)

class OhemCELoss2D(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self,
                 n_min,
                 thresh=0.7,
                 ignore_index=-1):

        super(OhemCELoss2D, self).__init__(None, None, ignore_index, reduction='none')
        
        self.thresh = -math.log(thresh)
        self.n_min = n_min
        self.ignore_index = ignore_index


    def forward(self, pred, target):
            return self.OhemCELoss(pred, target)
    

    def OhemCELoss(self, logits, labels):
        N, C, H, W = logits.size()
            
        loss = super(OhemCELoss2D, self).forward(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)
    


# knowledge distillation loss

# pixel-wise loss
class CriterionPixelWise(nn.Module):
    def __init__(self, ignore_index=250, use_weight=True, reduce='mean'):
        super(CriterionPixelWise, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduce)

    def forward(self, preds_S, preds_T):
 
        N,C,W,H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.permute(0,2,3,1).contiguous().view(-1,C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,1).contiguous().view(-1,C))))/W/H
        return loss


# pair-wise loss
# our model used scale=0.5, you can fix scale value
class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = self.sim_dis_compute
        self.scale = 0.5

    def forward(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        #feat_T.detach() # 필요
        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss

    def L2(self, f_):
        return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

    def similarity(self, feat):
        feat = feat.float()
        tmp = self.L2(feat) #.detach() # 필요
        feat = feat/tmp
        feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
        return torch.einsum('icm,icn->imn', [feat, feat])

    def sim_dis_compute(self, f_S, f_T):
        sim_err = ((self.similarity(f_T) - self.similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)
        sim_dis = sim_err.sum()
        return sim_dis
