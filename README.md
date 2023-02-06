# Making Pretrained Models of Color Recognition Module

Example of running the script for dataset called area3, train with resnet50, random weights, 3000 samples:
```bash
python3 main.py \
    --dataset sands8 \
    --num-labeled 20300 \
    --alpha 1.0 --lr 0.01 --batch-size 128 --aug-num 3 \
    --label-split 1 --progress True --epochs 100 \
    --eval-subdir="test" \
    --train-subdir="train" \
    --train_algo="vat" \
    --logdir=runs/pretrain/train_vat/rndweights_resnet10_norm_sands8_SGD \
    --experiment=rndweights
```

Example of testing and ploting confusion matrix for dataset called Area5
```bash
python3 eval_plot_cm_default.py \
    --dataset sands8 \
    --weights_path runs/pretrain/train_mixup_rampdown/initweights_resnet50_norm_sands8/state_dict_epoch_100.pth \
    --logdir runs_eval/pretrain_published_models
```


## Command line arguments

Look up https://github.com/kuanyewleong/coloured_image_classification/blob/main/config/cli.py for more details.

## Dataset Structure
We would organize datasets from different projects into files, different classes will go into different files. Something like the following structure:
        
        root/class_x/imagex.png
        root/class_x/imagey.png
        root/class_x/imagez.png

        root/class_y/imagex.png
        root/class_y/imagey.png
        root/class_y/imagez.png

Then during training, the module should progressively load the images (similar to the ImageNet data handling style).

So before running any training, make sure to prepare the data folders accordingly.

