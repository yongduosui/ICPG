# Environment
```shell
python==3.6
pytorch==1.7.0 
PyTorch Geometric
```

Install command:
```python
CUDA=cu101
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
pip install torch-geometric
```
# Command
Please refer to `run.sh`
