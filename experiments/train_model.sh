#### Train resnet20 on CIFAR10 160 epochs#####
python train_model.py \
  --pretrained_dataset cifar10 --pretrained_arch resnet20 \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 4 --ngpu 1

#### Train vgg16 on CIFAR10 160 epochs#####
python train_model.py \
  --pretrained_dataset cifar10 --pretrained_arch vgg16_cifar \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 4 --ngpu 1

#### Train resnet20 on CIFAR100 160 epochs#####
python train_model.py \
  --pretrained_dataset cifar100 --pretrained_arch resnet20 \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 4 --ngpu 1

#### Train vgg16 on CIFAR100 160 epochs#####
python train_model.py \
  --pretrained_dataset cifar100 --pretrained_arch vgg16_cifar \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 4 --ngpu 1

#### Train VGG16 on GTSRB 160 epochs #####
python train_model.py \
  --pretrained_dataset gtsrb --pretrained_arch vgg16_cifar \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 4 --ngpu 1

#### Train resnet20 on GTSRB 160 epochs #####
python train_model.py \
  --pretrained_dataset gtsrb --pretrained_arch resnet20 \
  --epochs 160 --batch_size 128 --learning_rate 0.1 --decay 1e-4 \
  --schedule 80 120 --gammas 0.1 0.1  --workers 4 --ngpu 1

