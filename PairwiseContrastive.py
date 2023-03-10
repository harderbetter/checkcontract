# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F


class PairwiseContrastive(nn.Module):
    def __init__(self):
        super(PairwiseContrastive, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        feats = F.normalize(feats, p=2, dim=1)
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            # pos_pair_ = torch.tensor(pos_pair_, dtype=torch.float)
            # pos_pair_ = pos_pair_[pos_pair_ <= 1.0]
            neg_pair_ = sim_mat[i][labels != labels[i]]
            # neg_pair_ = torch.tensor(neg_pair_, dtype=torch.float)

            if len(pos_pair_) < 1 or len(neg_pair_) < 1:
                continue

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True).cuda()

        loss = sum(loss) / batch_size
        return loss.type(torch.cuda.FloatTensor)
