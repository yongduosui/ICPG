dataset=NCI1

#ICPG
CUDA_VISIBLE_DEVICES=$1 \
python -u main_imp.py \
--dataset ${dataset} \
--get_mask_epochs 100 \
--epochs 100 \
--mask_lr 1e-2 \
--pruning_percent 0.05 \
--pruning_percent_w 0.2

# Random pruning
CUDA_VISIBLE_DEVICES=$1 python -u main_rp.py --dataset ${dataset} --random_type rprp --model GCN