CONFIG=$1
GPUS=$2
CUDA_VISIBLE_DEVICES=$GPUS python train.py --config $CONFIG