# Environment
```shell
python==3.6
pytorch==1.4.0 
PyTorch Geometric
```

Install command:
```python
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch # cuda101
CUDA=cu101
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-cluster==1.4.5 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-geometric==1.4.3
```
# Command
Please refer to `run.sh`
