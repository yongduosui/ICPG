
# ICPG
CUDA_VISIBLE_DEVICES=$1 python -u main_omp.py --save_dir omp_ckpt --idx 14 --mask_lr 0.001
# RP
CUDA_VISIBLE_DEVICES=$1 python -u main_rp.py --save_dir rp_ckpt