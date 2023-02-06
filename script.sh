##### --------------- r304 --------------------- #####

# python3 main.py \
#     --dataset r304 \
#     --num-labeled 3000 \
#     --alpha 1.0 --lr 0.01 --batch-size 128 --aug-num 3 \
#     --label-split 1 --progress True --epochs 100 \
#     --eval-subdir="valid" \
#     --train_algo="vat" \
#     --logdir=runs/pretrain/train_vat/rndweights_efficientnet_norm_304_small \
#     --experiment=rndweights

# python3 main.py \
#     --dataset r304 \
#     --num-labeled 3000 \
#     --alpha 1.0 --lr 0.001 --batch-size 128 --aug-num 3 \
#     --label-split 1 --progress True --epochs 5000 \
#     --eval-subdir="valid" \
#     --train_algo="vat" \
#     --logdir=runs/pretrain/train_vat/finetunefc_resnet50_norm_304_small_epoch5k \
#     --experiment=finetunefc

# python3 main.py \
#     --dataset r304 \
#     --num-labeled 3000 \
#     --alpha 1.0 --lr 0.01 --batch-size 128 --aug-num 3 \
#     --label-split 1 --progress True --epochs 100 \
#     --eval-subdir="valid" \
#     --train_algo="vat" \
#     --logdir=runs/pretrain/train_vat/initweights_resnet50_norm_304_small \
#     --experiment=initweights


# ##### --------------- area3 --------------------- #####

# python3 main.py \
#     --dataset area3 \
#     --num-labeled 3000 \
#     --alpha 1.0 --lr 0.01 --batch-size 128 --aug-num 3 \
#     --label-split 1 --progress True --epochs 100 \
#     --eval-subdir="valid" \
#     --train_algo="vat" \
#     --logdir=runs/pretrain/train_vat/rndweights_resnet50_norm_area3_small \
#     --experiment=rndweights

# python3 main.py \
#     --dataset area3 \
#     --num-labeled 3000 \
#     --alpha 1.0 --lr 0.001 --batch-size 128 --aug-num 3 \
#     --label-split 1 --progress True --epochs 5000 \
#     --eval-subdir="valid" \
#     --train_algo="vat" \
#     --logdir=runs/pretrain/train_vat/finetunefc_resnet50_norm_area3_small_epoch5k \
#     --experiment=finetunefc

# python3 main.py \
#     --dataset area3 \
#     --num-labeled 3000 \
#     --alpha 1.0 --lr 0.01 --batch-size 128 --aug-num 3 \
#     --label-split 1 --progress True --epochs 100 \
#     --eval-subdir="valid" \
#     --train_algo="vat" \
#     --logdir=runs/pretrain/train_vat/initweights_resnet50_norm_area3_small \
#     --experiment=initweights


# ##### --------------- area5 --------------------- #####

# python3 main.py \
#     --dataset area5 \
#     --num-labeled 3000 \
#     --alpha 1.0 --lr 0.01 --batch-size 128 --aug-num 3 \
#     --label-split 1 --progress True --epochs 100 \
#     --eval-subdir="valid" \
#     --train_algo="vat" \
#     --logdir=runs/pretrain/train_vat/rndweights_efficientnet_norm_area5_small \
#     --experiment=rndweights

# python3 main.py \
#     --dataset area5 \
#     --num-labeled 3000 \
#     --alpha 1.0 --lr 0.001 --batch-size 128 --aug-num 3 \
#     --label-split 1 --progress True --epochs 100 \
#     --eval-subdir="valid" \
#     --train_algo="vat" \
#     --logdir=runs/pretrain/train_vat/finetunefc_efficientnet_norm_area5_small \
#     --experiment=finetunefc

# python3 main.py \
#     --dataset area5 \
#     --num-labeled 3000 \
#     --alpha 1.0 --lr 0.01 --batch-size 128 --aug-num 3 \
#     --label-split 1 --progress True --epochs 100 \
#     --eval-subdir="valid" \
#     --train_algo="vat" \
#     --logdir=runs/pretrain/train_vat/initweights_efficientnet_norm_area5_small \
#     --experiment=initweights


python3 main.py \
    --dataset all_5datasets \
    --num-labeled 139200 \
    --alpha 1.0 --lr 0.01 --batch-size 128 --aug-num 3 \
    --label-split 1 --progress True --epochs 100 \
    --eval-subdir="test" \
    --train-subdir="train" \
    --train_algo="vat" \
    --logdir=runs/pretrain/train_vat/rndweights_resnet10_norm_all_5datasets_SGD \
    --experiment=rndweights
