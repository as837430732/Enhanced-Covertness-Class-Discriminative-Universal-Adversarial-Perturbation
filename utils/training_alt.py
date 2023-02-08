from __future__ import division
import numpy as np
import os, shutil, time
import itertools
import torch
import torch.nn.functional as F
import matplotlib
import copy
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.utils import time_string, print_log

def adjust_learning_rate(init_lr, init_momentum, optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr
    momentum = init_momentum
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
            momentum = momentum * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['momentum'] = momentum
    return lr

def alt_train(sources_data_loader, others_data_loader,
                    model, target_model, criterion_source,criterion_others, optimizer, epsilon, num_iterations, log,
                    print_freq=200, k=2, use_cuda=True, patch=False):
    # train function (forward, backward, update)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    source_losses = AverageMeter()
    others_losses = AverageMeter()
    source_top1 = AverageMeter()
    others_top1 = AverageMeter()
    source_top5 = AverageMeter()
    others_top5= AverageMeter()

    # switch to train mode
    model.module.generator.train()
    model.module.target_model.eval()
    target_model.eval()

    end = time.time()

    sources_data_iterator = iter(sources_data_loader)
    others_data_iterator = iter(others_data_loader)
    iteration = 0
    while (iteration < num_iterations ):

        for i in range(0,k):
            try:
                sources_input, sources_target = next(sources_data_iterator)  # 源类的输入和源类的标签
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                sources_data_iterator = iter(sources_data_loader)
                sources_input, sources_target = next(sources_data_iterator)

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                sources_target = sources_target.cuda()
                sources_input = sources_input.cuda()

            # compute output
            output = model(sources_input)  # UAP + targeted_model
            target_model_output = target_model(sources_input)

            source_loss = criterion_source(output, target_model_output, sources_target)

            # measure accuracy and record loss
            if len(sources_target.shape) > 1:
                sources_target = torch.argmax(sources_target, dim=-1)
            prec1, prec5 = accuracy(output.data, sources_target, topk=(1, 5))  # 添加扰动后的准确率
            source_losses.update(source_loss.item(), sources_input.size(0))
            source_top1.update(prec1.item(), sources_input.size(0))
            source_top5.update(prec5.item(), sources_input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            source_loss.backward()
            optimizer.step()

            # Project to l-infinity ball
            if patch:
                model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, 0, epsilon)
            else:
                model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, -epsilon, epsilon)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0:
            print_log('  Iteration: [{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                iteration, num_iterations, batch_time=batch_time,
                data_time=data_time, loss=source_losses, top1=source_top1, top5=source_top5) + time_string(),
                      log)

        for j in range(0, 1):
            try:
                others_input, others_target = next(others_data_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                others_data_iterator = iter(others_data_loader)
                others_input, others_target = next(others_data_iterator)

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                others_target = others_target.cuda()
                others_input = others_input.cuda()

            # compute output
            output = model(others_input)  # UAP + targeted_model
            target_model_output = target_model(others_input)
            others_loss = criterion_others(output, target_model_output, others_target)

            # measure accuracy and record loss
            if len(others_target.shape) > 1:
                others_target = torch.argmax(others_target, dim=-1)
            prec1, prec5 = accuracy(output.data, others_target, topk=(1, 5))  # 添加扰动后的准确率
            others_losses.update(others_loss.item(), others_input.size(0))
            others_top1.update(prec1.item(), others_input.size(0))
            others_top5.update(prec5.item(), others_input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            others_loss.backward()
            optimizer.step()

            # Project to l-infinity ball
            if patch:
                model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, 0, epsilon)
            else:
                model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, -epsilon, epsilon)

            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0:
            print_log('  Iteration: [{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                iteration, num_iterations, batch_time=batch_time,
                data_time=data_time, loss=others_losses, top1=others_top1, top5=others_top5) + time_string(),
                      log)


        iteration+=1


    print_log(
        '  **Train_source** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=source_top1,
                                                                                              top5=source_top5,
                                                                                              error1=100 - source_top1.avg),
        log)

    print_log(
        '  **Train_others** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=others_top1,
                                                                                              top5=others_top5,
                                                                                              error1=100 - others_top1.avg),
        log)

def train_target_model(train_loader, model, criterion, optimizer, epoch, log,
                       print_freq=200, use_cuda=True):
    # train function (forward, backward, update)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for iteration, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            target = target.cuda()
            input = input.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=-1)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, iteration, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                    top5=top5,
                                                                                                    error1=100 - top1.avg),
              log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log=None, use_cuda=True):
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):

        if use_cuda:
            target = target.cuda()
            input = input.cuda()

        with torch.no_grad():
            # compute output
            output = model(input)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
    if log:
        print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                       top5=top5,
                                                                                                       error1=100 - top1.avg),
                  log)
    else:
        print('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                   error1=100 - top1.avg))

    return top1.avg


def metrics_evaluate(source_loader, others_loader, target_model, perturbed_model, source_classes, sink_classes,
                     log=None, use_cuda=True):

    # switch to evaluate mode
    target_model.eval()
    perturbed_model.eval()

    for loader, loader_name in zip([source_loader, others_loader], ["Source", "Others"]):

        attack_success_rate = AverageMeter()  # Among the correctly classified samples, the ratio of being different from clean prediction (same as gt)
        if len(sink_classes) != 0:
            all_to_sink_success_rate = []
            source_to_sink_success_rate = []
            all_to_sink_success_rate_filtered = []
            for i in range(len(sink_classes)):
                all_to_sink_success_rate.append(AverageMeter())  # The ratio of samples going to the sink classes
                source_to_sink_success_rate.append(AverageMeter())
                all_to_sink_success_rate_filtered.append(AverageMeter())


        if len(loader) > 0:  # For UAP, all classes will be attacked, so others_loader is empty
            for input, gt in loader:  # gt:源类的标签
                if use_cuda:
                    gt = gt.cuda()
                    input = input.cuda()

                # compute output
                with torch.no_grad():
                    clean_output = target_model(input)  # [batch_size/2,num_classes]
                    pert_output = perturbed_model(input)  # [batch_size/2,num_classes]

                correctly_classified_mask = torch.argmax(clean_output, dim=-1).cpu() == gt.cpu()
                if torch.sum(correctly_classified_mask) > 0:
                    with torch.no_grad():
                        pert_output_corr_cl = perturbed_model(input[correctly_classified_mask])  # 只将分类正确的图像输入到网络中进行攻击
                    attack_succ_rate = accuracy(pert_output_corr_cl, gt[correctly_classified_mask], topk=(1,))
                    attack_success_rate.update(attack_succ_rate[0].item(), pert_output_corr_cl.size(0))

                # Collect samples from Source go to sink
                if len(sink_classes) != 0:
                    # Iterate over source class and sink class pairs
                    for cl_idx in range(len(sink_classes)):
                        source_cl = source_classes[0]
                        sink_cl = sink_classes
                        # 1. Check how many of the paired source class got to the sink class (Only relevant for source loader)
                        # Filter all idxs which belong to the source class
                        source_cl_idxs = [i == source_cl for i in gt]
                        source_cl_mask = torch.Tensor(source_cl_idxs) == True
                        if torch.sum(source_cl_mask) > 0:
                            gt_source_cl = gt[source_cl_mask]
                            pert_output_source_cl = pert_output[source_cl_mask]  # [batch_size, num_classes]

                            # Create desired target value
                            target_sink = torch.ones_like(gt_source_cl) * sink_cl[cl_idx]
                            source_to_sink_succ_rate = accuracy(pert_output_source_cl, target_sink, topk=(1,))
                            source_to_sink_success_rate[cl_idx].update(source_to_sink_succ_rate[0].item(),
                                                                       pert_output_source_cl.size(0))

                        # 2. How many of all samples go the sink class (Only relevant for others loader)
                        target_sink = torch.ones_like(gt) * sink_cl[cl_idx]
                        all_to_sink_succ_rate = accuracy(pert_output, target_sink, topk=(1,))
                        all_to_sink_success_rate[cl_idx].update(all_to_sink_succ_rate[0].item(), pert_output.size(0))

                        # 3. How many of all samples go the sink class, except gt sink class (Only relevant for others loader)
                        # Filter all idxs which are not belonging to sink class
                        non_sink_class_idxs = [i != sink_cl[cl_idx] for i in gt]
                        non_sink_class_mask = torch.Tensor(non_sink_class_idxs) == True
                        if torch.sum(non_sink_class_mask) > 0:
                            gt_non_sink_class = gt[non_sink_class_mask]
                            pert_output_non_sink_class = pert_output[non_sink_class_mask]

                            target_sink = torch.ones_like(gt_non_sink_class) * sink_cl[cl_idx]
                            all_to_sink_succ_rate_filtered = accuracy(pert_output_non_sink_class, target_sink,
                                                                      topk=(1,))
                            all_to_sink_success_rate_filtered[cl_idx].update(all_to_sink_succ_rate_filtered[0].item(),
                                                                             pert_output_non_sink_class.size(0))

            if log:
                print_log('\n\t########## {} #############'.format(loader_name), log)
                if len(sink_classes) != 0:
                    for cl_idx in range(len(sink_classes)):
                        source_cl = source_classes[0]
                        sink_cl = sink_classes
                        print_log('\n\tSource {} --> Sink {} Prec@1 {:.3f}'.format(source_cl, sink_cl[cl_idx],
                                                                                   source_to_sink_success_rate[
                                                                                       cl_idx].avg), log)
                        print_log(
                            '\tAll --> Sink {} Prec@1 {:.3f}'.format(sink_cl[cl_idx], all_to_sink_success_rate[cl_idx].avg),
                            log)
                        # Average fooling ratio of the non-source classes into the target label
                        # 非目标类别->sink类别的平均扰动率
                        print_log('\tAll (w/o sink samples) --> Sink {} Prec@1 {:.3f}'.format(sink_cl[cl_idx],
                                                                                              all_to_sink_success_rate_filtered[
                                                                                                  cl_idx].avg), log)
                if len(sink_classes) ==0:
                    print_log('\tPert cor Prec@1 {:.3f}'.format(attack_success_rate.avg), log)
                    print_log('\tf_r {:.3f}'.format(100 - attack_success_rate.avg), log)

            else:
                print('\n\t########## {} #############'.format(loader_name))
                if len(sink_classes) != 0:
                    for cl_idx in range(len(sink_classes)):
                        source_cl = source_classes[0]
                        sink_cl = sink_classes
                        print('\n\tSource {} --> Sink {} Prec@1 {:.3f}'.format(source_cl, sink_cl[cl_idx],
                                                                               source_to_sink_success_rate[cl_idx].avg))
                        print('\tAll --> Sink {} Prec@1 {:.3f}'.format(sink_cl[cl_idx], all_to_sink_success_rate[cl_idx].avg))
                        # Average fooling ratio of the non-source classes into the target label
                        print('\tAll (w/o sink samples) --> Sink {} Prec@1 {:.3f}'.format(sink_cl[cl_idx],
                                                                                          all_to_sink_success_rate_filtered[
                                                                                              cl_idx].avg))
                if len(sink_classes) ==0:
                    print_log('\tPert cor Prec@1 {:.3f}'.format(attack_success_rate.avg), log)
                    print_log('\tf_r {:.3f}'.format(100 - attack_success_rate.avg), log)


def save_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # 每一行取最大的前K个值的索引 [batch_size, 1]
        pred = pred.t()  # [1,batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [True,False,....] [1,batch_size]

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) # 正确个数
            res.append(correct_k.mul_(100.0 / batch_size))  #
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    #val: 当前batch的准确率 sum: 所有batch的准确率总和 avg: 平均准确率 n:batch_size
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_losses = self.epoch_losses - 1

        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = self.epoch_accuracy

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(
            self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1
        return self.max_accuracy(False) == val_acc

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain:
            return self.epoch_accuracy[:self.current_epoch, 0].max()
        else:
            return self.epoch_accuracy[:self.current_epoch, 1].max()

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis * 50, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis * 50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('---- save figure {} into {}'.format(title, save_path))
        plt.close(fig)
