GPU=$1
DATASET=cora
# best[32]: dim=256 lr=0.1 wd=0
# best[48]: 16 0.05 0
# best[9]: 128 0.1 5e-5
# best[12]: 128 0.1 0

# 512 0.01 5e-5
dim=512
lr=0.1
wd=1e-6
# dim=128
# lr=0.1 
# wd=1e-6
CUDA_VISIBLE_DEVICES=${GPU} python -u main_masker.py --dataset ${DATASET} --seed 618 --score_function concat_mlp \
--masker_dim ${dim} \
--mask_lr ${lr} \
--mask_wd ${wd}