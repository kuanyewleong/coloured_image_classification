# Making Pretrained Models of Color Recognition Module

Example of running the script for dataset called area3, train with resnet50, random weights, 3000 samples:
```bash
python3 main.py \
    --dataset area3 \
    --num-labeled 3000 \
    --alpha 1.0 --lr 0.01 --batch-size 128 --aug-num 3 \
    --label-split 1 --progress True --epochs 100 \
    --eval-subdir="valid" \
    --train_algo="vat" \
    --logdir=runs/pretrain/train_vat/rndweights_resnet50_norm_area3_small \
    --experiment=rndweights
```

Example of testing and ploting confusion matrix for dataset called Area5
```bash
python3 eval_plot_cm_default.py \
    --dataset area5 \
    --weights_path runs/pretrain/train_mixup_rampdown/initweights_resnet50_norm_area5_small/state_dict_epoch_100.pth \
    --logdir runs_eval/pretrain_published_models
```


## Command line arguments

Look up https://github.com/ais-research/pretrain_cr_model/blob/main/config/cli.py for more details.

## Dataset Structure
We would organize datasets from different projects into files, different classes will go into different files. Something like the following structure:
        
        root/class_x/imagex.png
        root/class_x/imagey.png
        root/class_x/imagez.png

        root/class_y/imagex.png
        root/class_y/imagey.png
        root/class_y/imagez.png

Then during training, the module should progressively load the images (similar to the ImageNet data handling style), this could overcome RAM limitation issue. This approach also can avoid the issue of old vs new styles / crop size / other differences used in creating those datasets by our team members in the past.

So before running any training, make sure to prepare the data folders accordingly.

