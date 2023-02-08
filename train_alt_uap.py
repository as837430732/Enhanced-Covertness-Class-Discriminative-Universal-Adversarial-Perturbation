from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
from itertools import cycle
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import OrderedDict

from torchvision import utils as vutils
from networks.uap import UAP
from utils.data import get_data_specs, get_data
from utils.utils import get_model_path, get_result_path, print_log
from utils.network import get_network, set_parameter_requires_grad

from utils.training_alt import alt_train,save_checkpoint,metrics_evaluate
from utils.custom_source_loss import LossConstructorSource
from utils.custom_others_loss import LossConstructorOthers


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a UAP')
    # pretrained
    parser.add_argument('--pretrained_dataset', default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb', 'imagenet'],
                        help='Used dataset (default: cifar10)')
    parser.add_argument('--pretrained_arch', default='resnet20',
                        choices=['resnet20', 'vgg16', 'resnet50','vgg16_cifar'],
                        help='Used model architecture: (default: resnet20)')
    parser.add_argument('--pretrained_seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')
    parser.add_argument('--finetune', action='store_true',
                        help='Finetune from pretrained imagenet weight (default: False)')

    # Parameters regarding UAP
    parser.add_argument('--epsilon', type=float, default=0.0392,
                        help='Norm restriction of UAP')  # L2: 7.843(2000/255) # Linf: 0.0392(10/255,像素值都映射到了[0,1])
    parser.add_argument('--source_classes', type=int, nargs='+', default=[],
                        help=' (default: [])')
    parser.add_argument('--sink_classes', type=int, nargs='+', default=[],
                        help=' (default: [])')
    parser.add_argument('--num_train_samples_per_class', type=int, default=-1,
                        help='Number of images to use (default: -1)')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='Number of iterations (default: 10)')
    parser.add_argument('--result_subfolder', default='default', type=str,
                        help='result subfolder name')

    # Optimization options
    parser.add_argument('--source_loss', default='bounded_logit_dec',
                        choices=['none', 'bounded_logit_dec','bounded_logit_source_sink', 'lt2'],
                        help='Used loss function for source classes: (default: bounded_logit_dec)')
    parser.add_argument('--others_loss', default='ce',
                        choices=['none', 'ce'],
                        help='Used loss function for other classes: (default: ce)')
    parser.add_argument('--confidence', default=0., type=float,
                        help='Confidence value for C&W losses (default: 0.0)')
    parser.add_argument('--k', default=2, type=int,
                        help='Training times on the targeted source image set (default: 2)')
    parser.add_argument('--lam', default=0.1, type=float,
                        help='The weighting value of logit pairing (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--print_freq', default=200, type=int, metavar='N',
                        help='print frequency (default: 200)')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of data loading workers (default: 6)')
    args = parser.parse_args()

    args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

    if args.pretrained_seed is None:
        args.pretrained_seed = random.randint(1, 10000)

    return args


def main():
    args = parse_arguments()

    random.seed(args.pretrained_seed)
    torch.manual_seed(args.pretrained_seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.pretrained_seed)
    cudnn.benchmark = True

    # Get data specs
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.pretrained_dataset, args.pretrained_arch)

    # Construct the array other classes:
    if args.pretrained_dataset in ["imagenet", "ycb"]:
        other_classes = args.source_classes
    else:
        all_classes = np.arange(num_classes) # [0 1 2 3 4 5 6 7 8 9]
        other_classes = [int(cl) for cl in all_classes if cl not in args.source_classes] # source_classes：源类， other_classes: 除源类的其他类

    # half_batch_size = args.batch_size // 2

    # get the result path to store the results
    result_path = get_result_path(dataset_name=args.pretrained_dataset,
                                  network_arch=args.pretrained_arch,
                                  random_seed=args.pretrained_seed,
                                  result_subfolder=args.result_subfolder,
                                  source_class=args.source_classes,
                                  sink_class=args.sink_classes)

    # Init logger
    log_file_name = os.path.join(result_path, 'log.txt')
    print("Log file: {}".format(log_file_name))
    log = open(log_file_name, 'w')
    print_log('save path : {}'.format(result_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        print_log("{} : {}".format(key, value), log)
    print_log("Random Seed: {}".format(args.pretrained_seed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("Torch  version : {}".format(torch.__version__), log)
    print_log("Cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    # data_train_sources:train数据集中标签为source_classes的数据
    data_train_sources, data_test_sources = get_data(args.pretrained_dataset,
                                                     mean=mean,
                                                     std=std,
                                                     input_size=input_size,
                                                     classes=args.source_classes,
                                                     train_samples_per_class=args.num_train_samples_per_class)
    data_train_sources_loader = torch.utils.data.DataLoader(data_train_sources,
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=args.workers,
                                                            pin_memory=True)

    data_test_sources_loader = torch.utils.data.DataLoader(data_test_sources,
                                                           batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=args.workers,
                                                           pin_memory=True)

    data_train_others, data_test_others = get_data(args.pretrained_dataset,
                                                   mean=mean,
                                                   std=std,
                                                   input_size=input_size,
                                                   classes=other_classes,
                                                   others=True,
                                                   train_samples_per_class=args.num_train_samples_per_class)
    data_train_others_loader = torch.utils.data.DataLoader(data_train_others,
                                                           batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.workers,
                                                           pin_memory=True)

    data_test_others_loader = torch.utils.data.DataLoader(data_test_others,
                                                          batch_size=args.batch_size,
                                                          shuffle=False,
                                                          num_workers=args.workers,
                                                          pin_memory=True)

    # Init model, criterion, and optimizer
    print_log("=> Creating model '{}'".format(args.pretrained_arch), log)
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.pretrained_dataset,
                                network_arch=args.pretrained_arch,
                                random_seed=args.pretrained_seed)
    model_weights_path = os.path.join(model_path, "checkpoint.pth.tar")

    target_network = get_network(args.pretrained_arch, input_size=input_size, num_classes=num_classes,
                                 finetune=args.finetune)
    # print_log("=> Network :\n {}".format(target_network), log)
    target_network = torch.nn.DataParallel(target_network, device_ids=list(range(args.ngpu)))
    # Set the target model into evaluation mode
    target_network.eval()
    # Imagenet models use the pretrained pytorch weights
    if args.pretrained_dataset != "imagenet":
        network_data = torch.load(model_weights_path)
        target_network.load_state_dict(network_data['state_dict'])

    # Set all weights to not trainable
    set_parameter_requires_grad(target_network, requires_grad=False)



    print_log("=> Inserting Generator", log)
    generator = UAP(shape=(input_size, input_size),
                    num_channels=num_channels,
                    mean=mean,
                    std=std,
                    use_cuda=args.use_cuda)

    print_log("=> Generator :\n {}".format(generator), log)


    # pertubed_net：生成UAP的模型+目标模型
    perturbed_net = nn.Sequential(OrderedDict([('generator', generator), ('target_model', target_network)]))
    perturbed_net = torch.nn.DataParallel(perturbed_net, device_ids=list(range(args.ngpu)))


    # Set the target model into evaluation mode
    perturbed_net.module.target_model.eval()
    perturbed_net.module.generator.train()

    criterion_source = LossConstructorSource(source_classes=args.source_classes,
                                sink_classes=args.sink_classes,
                                num_classes=num_classes,
                                source_loss=args.source_loss,
                                confidence=args.confidence,
                                use_cuda=args.use_cuda)
    criterion_others = LossConstructorOthers(source_classes=args.source_classes,
                                             sink_classes=args.sink_classes,
                                             num_classes=num_classes,
                                             others_loss=args.others_loss,
                                             lam=args.lam,
                                             use_cuda=args.use_cuda)

    if args.use_cuda:
        target_network.cuda()
        generator.cuda()
        perturbed_net.cuda()
        criterion_source.cuda()
        criterion_others.cuda()

    optimizer = torch.optim.Adam(perturbed_net.parameters(),
                                 lr=state['learning_rate'], weight_decay=0.01)  # betas=(0.5, 0.999)



    start_time = time.time()
    alt_train(sources_data_loader=data_train_sources_loader,
                    others_data_loader=data_train_others_loader,
                    model=perturbed_net,
                    target_model=target_network,
                    criterion_source=criterion_source,
                    criterion_others=criterion_others,
                    optimizer=optimizer,
                    epsilon=args.epsilon,
                    num_iterations=args.num_iterations,
                    log=log,
                    print_freq=args.print_freq,
                    k=args.k,
                    use_cuda=args.use_cuda)
    end_time = time.time()
    print_log("Elapsed generation time: {}".format(end_time - start_time), log)

    # evaluate
    print_log("Final evaluation:", log)
    metrics_evaluate(source_loader=data_test_sources_loader,
                     others_loader=data_test_others_loader,
                     target_model=target_network,
                     perturbed_model=perturbed_net,
                     source_classes=args.source_classes,
                     sink_classes=args.sink_classes,
                     log=log,
                     use_cuda=args.use_cuda)

    # save_checkpoint({
    #     'arch': args.pretrained_arch,
    #     'state_dict': perturbed_net.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'args': copy.deepcopy(args),
    # }, result_path, 'checkpoint.pth.tar')

    # Plot the adversarial perturbation
    uap_numpy = perturbed_net.module.generator.uap.detach().cpu().numpy()

    uap_file_name = os.path.join(result_path, 'uap.png')
    vutils.save_image(perturbed_net.module.generator.uap.detach().cpu(), uap_file_name)

    # Calculate the norm
    uap_norm = np.linalg.norm(uap_numpy.reshape(-1), np.inf)
    print_log("Norm of UAP: {}".format(uap_norm), log)

    log.close()


if __name__ == '__main__':
    main()
