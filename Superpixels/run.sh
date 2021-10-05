GPU=$1
seed=41
code=main_imp.py 
dataset=MNIST

CUDA_VISIBLE_DEVICES=$1 \
python -u $code --config 'configs/superpixels_graph_classification_GCN_MNIST_100k.json' \
--dataset $dataset \
--seed $seed  \
--mask_epochs 100 \
--eval_epochs 100