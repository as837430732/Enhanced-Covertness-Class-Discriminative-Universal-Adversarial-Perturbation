import numpy as np
import torch
import random

from random import randint
from torch.nn.modules.loss import _WeightedLoss
from utils.utils import one_hot


class LossConstructorSource(_WeightedLoss):
    def __init__(self, source_classes, sink_classes, num_classes, source_loss,
                 weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='none',
                 confidence=0.0,
                 use_cuda=True):
        super(LossConstructorSource, self).__init__(weight, size_average, reduce, reduction)
        assert len(source_classes) >= 1

        self.source_classes = source_classes
        self.sink_classes = sink_classes
        self.num_classes = num_classes
        self.use_cuda = use_cuda
        all_classes = np.arange(num_classes)
        self.other_classes = [cl for cl in all_classes if cl not in source_classes]
        self.confidence = confidence


        # Select source loss:
        if source_loss == "none":
            self.source_loss_fn = empty_loss
        elif source_loss == "bounded_logit_dec":
            self.source_loss_fn = bounded_logit_dec
        elif source_loss == "bounded_logit_source_sink":
            self.source_loss_fn = bounded_logit_source_sink
        elif source_loss == "lt2":
            self.source_loss_fn = lt2
        else:
            raise ValueError()



    def forward(self, perturbed_logit, clean_logit, gt):
        # Consider only sample that are correctly classified
        clean_class = torch.argmax(clean_logit, dim=-1)
        correct_cl_mask = clean_class == gt
        perturbed_logit = perturbed_logit[correct_cl_mask]
        clean_class = clean_class[correct_cl_mask]


        source_classes_idxs = [i in self.source_classes for i in clean_class]
        source_classes_mask = torch.Tensor(source_classes_idxs) == True
        if torch.sum(source_classes_mask) > 0:
            perturbed_logit_source = perturbed_logit[source_classes_mask]
            clean_class_source = clean_class[source_classes_mask]  # batch_size/2 - 没有正确分类的
            # 原始类别为源类的图像添加扰动后输入目标模型的logit结果，正确分类的源类的类别

            source_loss = self.source_loss_fn(perturbed_logit=perturbed_logit_source, clean_class=clean_class_source,
                                              source_classes=self.source_classes,
                                              sink_classes=self.sink_classes,
                                              num_classes=self.num_classes,
                                              confidence=self.confidence,
                                              use_cuda=self.use_cuda)
        else:
            source_loss = torch.tensor([])
            if self.use_cuda:
                source_loss = source_loss.cuda()



        loss = source_loss

        if len(loss) == 0:
            loss = torch.tensor([0.], requires_grad=True)
        return torch.mean(loss)


def empty_loss(perturbed_logit, clean_class, source_classes, sink_classes, num_classes=-1, confidence=0.0,
               use_cuda=False):
    loss = torch.tensor([], requires_grad=True)
    if use_cuda:
        loss = loss.cuda()
    return loss




# 针对目标类别，增加非原始类别中最大的类别logit值，直至大于目标类别图像的原始类别，论文中的L_BL_t
def bounded_logit_dec(perturbed_logit, clean_class, source_classes, sink_classes, num_classes, confidence=0.0,
                      use_cuda=False):
    one_hot_labels = one_hot(clean_class.cpu(), num_classes=num_classes)

    # perturbed_logit = torch.softmax(perturbed_logit, dim=1)
    if use_cuda:
        one_hot_labels = one_hot_labels.cuda()

    class_logits = (one_hot_labels * perturbed_logit).sum(1)
    not_class_logits = ((1. - one_hot_labels) * perturbed_logit - one_hot_labels * 10000.).max(1)[0]

    # class_logits = torch.log(class_logits)
    # not_class_logits = torch.log(not_class_logits)

    loss = torch.clamp(class_logits - not_class_logits, min=-confidence)
    return loss


# 针对目标类别，论文中的Lt1和Lt2
def bounded_logit_source_sink(perturbed_logit, clean_class, source_classes, sink_classes, num_classes, confidence=0.0,
                              use_cuda=False):
    one_hot_labels = one_hot(clean_class.cpu(), num_classes=num_classes)  # 源类的one-hot表示
    if use_cuda:
        one_hot_labels = one_hot_labels.cuda()

    # perturbed_logit = torch.softmax(perturbed_logit, dim=1)

    loss = torch.tensor([])
    if use_cuda:
        loss = loss.cuda()

    source_cl = source_classes[0]
    sink_cl = sink_classes
    # Filter all idxs which belong to the source class
    source_cl_idxs = [i == source_cl for i in clean_class]
    source_cl_mask = torch.Tensor(source_cl_idxs) == True
    if torch.sum(source_cl_mask) > 0:
        clean_class_source_cl = clean_class[source_cl_mask]
        one_hot_labels_source_cl = one_hot_labels[source_cl_mask]
        perturbed_logit_source_cl = perturbed_logit[source_cl_mask]

        # source loss: Decrease the Source part
        class_logits_source_cl = (one_hot_labels_source_cl * perturbed_logit_source_cl).sum(1)
        not_class_logits_source_cl = ((1. - one_hot_labels_source_cl) * perturbed_logit_source_cl - one_hot_labels_source_cl * 10000.).max(1)[0].detach()  # 除了源类最大类别的logit值
        # source_cl_loss = torch.clamp(class_logits_source_cl - not_class_logits_source_cl, min=-confidence)

        # class_logits_source_cl = torch.log(class_logits_source_cl)
        # not_class_logits_source_cl = torch.log(not_class_logits_source_cl)

        source_cl_loss = torch.clamp(class_logits_source_cl - not_class_logits_source_cl, min=0)

        random.seed()
        rand = randint(0, len(sink_cl)-1)
        # print("sink_cl ",sink_cl)
        # print("sink_cl ",sink_cl[rand])

        # sink loss: Increase the Sink part
        target_sink_class = torch.ones_like(clean_class_source_cl) * sink_cl[rand]
        one_hot_labels_sink_cl = one_hot(target_sink_class.cpu(), num_classes=num_classes)  # sink类的One-hot表示
        if use_cuda:
            one_hot_labels_sink_cl = one_hot_labels_sink_cl.cuda()
        class_logits_sink_cl = (one_hot_labels_sink_cl * perturbed_logit_source_cl).sum(1)  # [64,10]->[64] 扰动后sink类别的logit值
        not_class_logits_sink_cl = ((1. - one_hot_labels_sink_cl) * perturbed_logit_source_cl - one_hot_labels_sink_cl * 10000.).max(1)[0].detach() #非sink类别的最大类别的logit值

        # class_logits_sink_cl = torch.log(class_logits_sink_cl)
        # not_class_logits_sink_cl = torch.log(not_class_logits_sink_cl)

        sink_cl_loss = torch.clamp(not_class_logits_sink_cl - class_logits_sink_cl, min=-confidence)

        loss_source_cl = (source_cl_loss + sink_cl_loss) / 2.

        loss = torch.cat((loss, loss_source_cl), 0)

    assert len(loss) == len(clean_class)  # Can be deleted after a few tries

    return loss




# 针对目标类别，论文中的lt2
def lt2(perturbed_logit, clean_class, source_classes, sink_classes, num_classes, confidence=0.0, use_cuda=False):
    one_hot_labels = one_hot(clean_class.cpu(), num_classes=num_classes)
    if use_cuda:
        one_hot_labels = one_hot_labels.cuda()

    loss = torch.tensor([])
    if use_cuda:
        loss = loss.cuda()
    # perturbed_logit = torch.softmax(perturbed_logit, dim=1)

    for source_cl, sink_cl in zip(source_classes, sink_classes):
        # Filter all idxs which belong to the source class
        source_cl_idxs = [i == source_cl for i in clean_class]
        source_cl_mask = torch.Tensor(source_cl_idxs) == True
        if torch.sum(source_cl_mask) > 0:
            clean_class_source_cl = clean_class[source_cl_mask]
            one_hot_labels_source_cl = one_hot_labels[source_cl_mask]
            perturbed_logit_source_cl = perturbed_logit[source_cl_mask]

            # sink loss: Increase the Sink part
            target_sink_class = torch.ones_like(clean_class_source_cl) * sink_cl
            one_hot_labels_sink_cl = one_hot(target_sink_class.cpu(), num_classes=num_classes)
            if use_cuda:
                one_hot_labels_sink_cl = one_hot_labels_sink_cl.cuda()
            class_logits_sink_cl = (one_hot_labels_sink_cl * perturbed_logit_source_cl).sum(1)
            not_class_logits_sink_cl = \
            ((1. - one_hot_labels_sink_cl) * perturbed_logit_source_cl - one_hot_labels_sink_cl * 10000.).max(1)[
                0].detach()
            # class_logits_sink_cl = torch.log(class_logits_sink_cl)
            # not_class_logits_sink_cl = torch.log(not_class_logits_sink_cl)
            sink_cl_loss = torch.clamp(not_class_logits_sink_cl - class_logits_sink_cl, min=-confidence)
            #print("class_logits_sink_cl",class_logits_sink_cl)
            #print("not_class_logits_sink_cl",not_class_logits_sink_cl)
            #print("sink_cl_loss",sink_cl_loss)

            loss_source_cl = sink_cl_loss

            loss = torch.cat((loss, loss_source_cl), 0)

    return loss
