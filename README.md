## An Empirical Study of Deep Graph Neural Networks



### Requirements

Environments: Xeon Platinum 8255C (CPU), 384GB (RAM), Tesla V100 32GB (GPU), Ubuntu 16.04 (OS).

The PyTorch version we use is torch 1.7.1+cu110. Please refer to the official website -- https://pytorch.org/get-started/locally/ -- for the detailed installation instructions.

We also use PyG to preprocess the data in the ogbn-papers100M dataset. Please refer to the official website -- https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html -- for the detailed installation instructions.

To install other requirements:

```setup
pip install -r requirements.txt
```

 ### SGC+AIR Training:

To reproduce the experimental results on the three small citation networks, please run the following commands:

```bash
cd ./src

python sgc_air.py --dataset cora --lr 0.1 --weight_decay 5e-4 --hidden 200 --dropout 0.4 --hops 10
python sgc_air.py --dataset citeseer --lr 0.1 --weight_decay 1e-3 --hidden 200 --dropout 0.2 --hops 15
python sgc_air.py --dataset pubmed --lr 0.05 --weight_decay 5e-4 --hidden 200 --dropout 0.5 --hops 30
```

For the three OGB dataset, please run the following commands:

```bash
cd ./src

```

 ### APPNP+AIR Training:

To reproduce the experimental results on the three small citation networks, please run the following commands:

```bash
cd ./src

python appnp_air.py --dataset cora --lr 0.1 --weight_decay 3e-3 --hidden 200 --dropout 0.2 --alpha 0.95 --hops 10
python appnp_air.py --dataset citeseer --lr 0.01 --weight_decay 3e-3 --hidden 200 --dropout 0.2 --alpha 0.95 --hops 10
python apppnp_air.py --dataset pubmed --lr 0.05 --weight_decay 5e-4 --hidden 200 --dropout 0.5 --alpha 0.95 --hops 10
```

For the ogbn-arxiv dataset, please run the following commands:

```bash
cd ./src

```

 ### GCN+AIR Training:

To reproduce the experimental results on the three small citation networks, please run the following commands:

```bash
cd ./src

python gcn_air.py --dataset cora --lr 0.01 --weight_decay 1e-3 --hidden 32 --dropout 0.5 --hops 6
python gcn_air.py --dataset citeseer --lr 0.01 --weight_decay 1e-2 --hidden 16 --dropout 0.3 --hops 4
python gcn_air.py --dataset pubmed --lr 0.1 --weight_decay 1e-3 --hidden 32 --dropout 0.5 --hops 4
```

For the ogbn-arxiv dataset, please run the following commands:

```bash
cd ./src

```



### Node Classification Results:

<img src=".\node_classifi_perf.png" style="zoom:80%;" />
