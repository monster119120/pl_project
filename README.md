CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ARCH=resnet50 python main.py --gpus 6 --lr 0.001 --data-path ../data/imagenet --pretrained
