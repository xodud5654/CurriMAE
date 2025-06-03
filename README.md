# CurriMAE

## Paper
 TBD
# Pre-training

## Self-supervised with Chest X-rays (ViTs)
We pretrain ViTs with MAE following the official repo but with **a customized recipe **(please refer to the paper for more details). Two sample commands are provided below.

###### To pretrain curriMAE based on ViT-S on CheXpert and CXR8:

```
Python --use_env main_pretrain_multi_datasets_xray.py \
 --output_dir ${SAVE_DIR} \
 --log_dir ${SAVE_DIR} \
 --batch_size 256 \
 --model mae_vit_small_patch16 \
 --mask_ratio 0.6 \
 --curri True
 --epochs 800 \
 --warmup_epochs 40 \
 --blr 1.5e-4 --weight_decay 0.05 \
 --num_workers 8 \
 --input_size 224 \
 --mask_strategy 'random' \
 --random_resize_range 0.5 1.0 \
 --datasets_names chexpert
```



##### Fine-tune ViTs

```
python --use_env main_finetune_chestxray.py \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --batch_size 128 \
    --finetune "CurriMAE-pretrain_800E.pth" \
    --epochs 75 \
    --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 \
    --model vit_small_patch16 \
    --warmup_epochs 5 \
    --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path ${DATASET_DIR} \
    --num_workers 12 \
    --train_list ${TRAIN_LIST} \
    --val_list ${VAL_LIST} \
    --test_list ${TEST_LIST} \
    --nb_classes 6  \
    --eval_interval 10 \
    --min_lr 1e-5 \
    --build_timm_transform \
    --aa 'rand-m6-mstd0.5-inc1'
    -- best_model ‘loss’ \
    -- data_type ‘multi-label’ \
    -- average ‘weighted’ \
```

## Citation

```
@inproceedings{xiao2023delving,
  title={Delving into masked autoencoders for multi-label thorax disease classification},
  author={Xiao, Junfei and Bai, Yutong and Yuille, Alan and Zhou, Zongwei},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3588--3600},
  year={2023}
}
```
![image](https://github.com/user-attachments/assets/6cb89caf-5df6-41a9-abae-34fe7fc42ceb)
