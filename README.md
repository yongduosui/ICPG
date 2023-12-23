# Inductive Lottery Ticket Learning for Graph Neural Networks

We provide a detailed code for "Inductive Lottery Ticket Learning for Graph Neural Networks".

Yongduo Sui, Xiang Wang, Tianlong Chen, Meng Wang, Xiangnan He, Tat-Seng Chua.

In JCST 2023 (Journal of Computer Science and Technology): https://jcst.ict.ac.cn/en/article/doi/10.1007/s11390-023-2583-5 



## Environment
```shell
python==3.6
pytorch==1.7.1
PyTorch Geometric
```
Install command:
```python
CUDA=cu101
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+${CUDA}.html
pip install torch-geometric
```

 
## Codes for ICPG
 For different datasets please refer to TUDataset, Superpixels, OGB and NodeClassification folders.
