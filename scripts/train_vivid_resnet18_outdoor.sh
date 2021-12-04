DATA_ROOT=/HDD/Dataset_processed
TRAIN_SET=$DATA_ROOT/VIVID_256
GPU_NUM=0

CUDA_VISIBLE_DEVICES=${GPU_NUM} \
python train.py $TRAIN_SET \
--resnet-layers 18 \
--num-scales 1 \
--scene_type outdoor \
-b 4 -s 0 -t 1.0 -r 0.1 \
--rgb-ssim 0.30 --thr-ssim 0.85 \
--epoch-size 400 --sequence-length 3 \
--with-ssim 1 \
--with-thr-mask 1 \
--with-rgb-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output --with-gt \
--name vivid_resnet18_outdoor
