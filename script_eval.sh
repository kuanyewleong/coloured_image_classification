# python3 eval_plot_cm_default.py \
#     --dataset area5 \
#     --weights_path runs/pretrain/train_mixup_rampdown/finetunefc_efficientnet_norm_area5_small/state_dict_epoch_100.pth \
#     --logdir runs_eval/pretrain_published_models/mixup \
#     --experiment=finetunefc \
#     --train_algo=mixup


# python3 eval_plot_cm_default.py \
#     --dataset area5 \
#     --weights_path runs/pretrain/train_mixup_rampdown/initweights_efficientnet_norm_area5_small/state_dict_epoch_100.pth \
#     --logdir runs_eval/pretrain_published_models/mixup \
#     --experiment=initweights \
#     --train_algo=mixup


python3 eval_plot_cm_default.py \
    --dataset all_4datasets \
    --weights_path runs/pretrain/train_vat/rndweights_efficientnet_norm_all_4datasets_SGD/state_dict_epoch_100.pth \
    --logdir runs_eval/pretrained_angel_cr_models \
    --experiment=rndweights \
    --train_algo=vat \
    --eval-subdir=test

# python3 eval_plot_cm_mixup_vat.py \
#     --dataset area5 \
#     --weights_path runs/pretrain/train_mixup_vat/finetunefc_efficientnet_norm_area5_small/state_dict_epoch_100.pth \
#     --logdir runs_eval/pretrain_published_models/mixup_vat \
#     --experiment=finetunefc \
#     --train_algo="mixup+vat"


# python3 eval_plot_cm_mixup_vat.py \
#     --dataset area5 \
#     --weights_path runs/pretrain/train_mixup_vat/initweights_efficientnet_norm_area5_small/state_dict_epoch_100.pth \
#     --logdir runs_eval/pretrain_published_models/mixup_vat \
#     --experiment=initweights \
#     --train_algo="mixup+vat"


# python3 eval_plot_cm_mixup_vat.py \
#     --dataset area5 \
#     --weights_path runs/pretrain/train_mixup_vat/rndweights_efficientnet_norm_area5_small/state_dict_epoch_100.pth \
#     --logdir runs_eval/pretrain_published_models/mixup_vat \
#     --experiment=rndweights \
#     --train_algo="mixup+vat"