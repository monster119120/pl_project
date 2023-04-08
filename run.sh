CUDA_VISIBLE_DEVICES=7 python main.py --arch resnet50 --pretrained --data-path ../data/imagenet -e \
    --accelerator gpu --devices 1
