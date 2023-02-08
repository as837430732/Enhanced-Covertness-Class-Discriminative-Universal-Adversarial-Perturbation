import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class LossConstructorOthers(_WeightedLoss):
    def __init__(self, source_classes, sink_classes, num_classes, others_loss,
                 weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='none', lam=0.1,
                 use_cuda=True):
        super(LossConstructorOthers, self).__init__(weight, size_average, reduce, reduction)
        assert len(source_classes) >= 1

        self.source_classes = source_classes
        self.sink_classes = sink_classes
        self.num_classes = num_classes
        self.use_cuda = use_cuda
        all_classes = np.arange(num_classes)
        self.other_classes = [cl for cl in all_classes if cl not in source_classes]
        self.lam = lam

        # Select others loss:
        if others_loss == "none":
            self.others_loss_fn = empty_loss
        elif others_loss == "ce_pair":
            self.others_loss_fn = ce_pair
        else:
            raise ValueError()

    def forward(self, perturbed_logit, clean_logit, gt):
        # Consider only sample that are correctly classified
        clean_class = torch.argmax(clean_logit, dim=-1)
        correct_cl_mask = clean_class == gt
        perturbed_logit = perturbed_logit[correct_cl_mask]
        clean_class = clean_class[correct_cl_mask]

        source_classes_idxs = [i in self.source_classes for i in clean_class]
        other_classes_idxs = (~np.array(source_classes_idxs)).tolist()
        other_classes_mask = torch.Tensor(other_classes_idxs) == True
        if torch.sum(other_classes_mask) > 0:
            perturbed_logit_others = perturbed_logit[other_classes_mask]
            clean_class_others = clean_class[other_classes_mask]
            others_loss = self.others_loss_fn(perturbed_logit=perturbed_logit_others,
                                              clean_logit=clean_logit,
                                              clean_class=clean_class_others,
                                              source_classes=self.source_classes,
                                              sink_classes=self.sink_classes,
                                              num_classes=self.num_classes,
                                              lam=self.lam,
                                              use_cuda=self.use_cuda)
        else:
            others_loss = torch.tensor([])
            if self.use_cuda:
                others_loss = others_loss.cuda()

        # loss = torch.log(others_loss)
        loss = others_loss
        if len(loss) == 0:
            loss = torch.tensor([0.], requires_grad=True)
        return torch.mean(loss)


def empty_loss(perturbed_logit, clean_class, source_classes, sink_classes, num_classes=-1, use_cuda=False):
    loss = torch.tensor([], requires_grad=True)
    if use_cuda:
        loss = loss.cuda()
    return loss


# 针对非目标类别，论文中的Lnt
def ce_pair(perturbed_logit, clean_logit, clean_class, source_classes, sink_classes, num_classes=-1, lam=0.1,
       use_cuda=False):
    if len(sink_classes) > 0:
        pass
    # perturbed_logit= torch.softmax(perturbed_logit,dim=1)
    ce_loss = F.cross_entropy(perturbed_logit, clean_class,
                              weight=None, ignore_index=-100, reduction='none')
    logit_pair_loss = F.mse_loss(perturbed_logit, clean_logit)

    loss = ce_loss + lam * logit_pair_loss
    return loss
