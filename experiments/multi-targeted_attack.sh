#### Training perturbation on GTSRB (resnet20) #####
python train_alt_uap.py \
      --pretrained_dataset GTSRB --pretrained_arch resnet20 \
      --epsilon 0.05882 \
      --source_loss bounded_logit_source_sink --others_loss ce_pair --confidence 10 \
      --num_iterations 500 \
      --batch_size 128 --learning_rate 0.005\
      --source_classes 9 \
      --sink_classes 4 8 \
      --workers 4 --ngpu 1 \
      --result_subfolder multi-targeted_attack

#### Training perturbation on GTSRB (vgg16) #####
python train_alt_uap.py \
      --pretrained_dataset GTSRB --pretrained_arch vgg16_cifar \
      --epsilon 0.05882 \
      --source_loss bounded_logit_source_sink --others_loss ce_pair --confidence 10 \
      --num_iterations 500 \
      --batch_size 128 --learning_rate 0.005\
      --source_classes 9 \
      --sink_classes 4 8 \
      --workers 4 --ngpu 1 \
      --result_subfolder multi-targeted_attack