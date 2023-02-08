#
#### Training perturbation on CIFAR10 (resnet20) #####
python train_alt_uap.py \
      --pretrained_dataset cifar10 --pretrained_arch resnet20 \
      --epsilon 0.05882 \
      --source_loss bounded_logit_source_sink --others_loss ce_pair --confidence 10 \
      --num_iterations 500 \
      --batch_size 128 --learning_rate 0.005\
      --source_classes 2 \
      --sink_classes 0 \
      --workers 4 --ngpu 1 \
      --result_subfolder one-targeted_attack

#### Training perturbation on CIFAR10 (vgg16) #####
python train_alt_uap.py \
      --pretrained_dataset cifar10 --pretrained_arch vgg16_cifar \
      --epsilon 0.05882 \
      --source_loss bounded_logit_source_sink --others_loss ce_pair --confidence 10 \
      --num_iterations 500 \
      --batch_size 128 --learning_rate 0.005\
      --source_classes 2 \
      --sink_classes 0 \
      --workers 4 --ngpu 1 \
      --result_subfolder one-targeted_attack

#### Training perturbation on CIFAR100 (resnet20) #####
python train_alt_uap.py \
      --pretrained_dataset cifar100 --pretrained_arch resnet20 \
      --epsilon 0.05882 \
      --source_loss bounded_logit_source_sink --others_loss ce_pair --confidence 10 \
      --num_iterations 500 \
      --batch_size 128 --learning_rate 0.005\
      --source_classes 2 \
      --sink_classes 0 \
      --workers 4 --ngpu 1 \
      --result_subfolder one-targeted_attack

#### Training perturbation on CIFAR100 (vgg16) #####
python train_alt_uap.py \
      --pretrained_dataset cifar100 --pretrained_arch vgg16_cifar \
      --epsilon 0.05882 \
      --source_loss bounded_logit_source_sink --others_loss ce_pair --confidence 10 \
      --num_iterations 500 \
      --batch_size 128 --learning_rate 0.005\
      --source_classes 2 \
      --sink_classes 0 \
      --workers 4 --ngpu 1 \
      --result_subfolder one-targeted_attack

#### Training perturbation on imagenet (resnet50)#####
python train_alt_uap.py \
      --pretrained_dataset imagenet --pretrained_arch resnet50 \
      --epsilon 0.05882 \
      --source_loss bounded_logit_source_sink --others_loss ce_pair --confidence 10 \
      --num_iterations 500 \
      --batch_size 128 --learning_rate 0.005\
      --source_classes 2 \
      --sink_classes 10 \
      --workers 4 --ngpu 1 \
      --result_subfolder one-targeted_attack

#### Training perturbation on imagenet (vgg16)#####
python train_alt_uap.py \
      --pretrained_dataset imagenet --pretrained_arch vgg16 \
      --epsilon 0.05882 \
      --source_loss bounded_logit_source_sink --others_loss ce_pair --confidence 10 \
      --num_iterations 500 \
      --batch_size 128 --learning_rate 0.005\
      --source_classes 2 \
      --sink_classes 10 \
      --workers 4 --ngpu 1 \
      --result_subfolder one-targeted_attack