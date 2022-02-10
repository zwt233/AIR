## An Empirical Study of Deep Graph Neural Networks



### Requirements

Environments: Xeon Platinum 8255C (CPU), 384GB (RAM), Tesla V100 32GB (GPU), Ubuntu 16.04 (OS).

The PyTorch version we use is torch 1.7.1+cu101. Please refer to the official website -- https://pytorch.org/get-started/locally/ -- for the detailed installation instructions.

To install all the requirements:

```setup
pip install -r requirements.txt
```



### Experimental Analysis

We implement **ResGCN**, **DenseGCN**, **MLP+Res**, **MLP+Dense**, **SGC**, and **2 GCN variants** on our own in 

```
./src/models.py
```



The code of **ResGCN** and **DenseGCN** is in 

```
./src/gcn_sc.py
```

The code of **MLP+Res** and **MLP+Dense** is in 

```
./src/mlp_sc.py
```

The code of **SGC** is in 

```
./src/sgc.py
```

The code of **GCN with D<sub>t</sub>=2** and **GCN with D<sub>p</sub>=2D<sub>t</sub>** is in 

```
./src/gcn_2dt.py,  ./src/gcn_dp2dt.py
```

The code for printing the gradient of the first layer of GCN is in 

```
./src/print_gradient.py
```

The code for the scalability experiment is provided in 

```
./src/scalability/
```

please run gen_graph.py first to generate artificial graphs; 

then run 

```
./src/scalability/appnp/gcn/dgmlp.py --n="graph_size"
```

where "graph size" varies from 100,000 to 1,000,000 with the step of 100,000.

We also provide the official code of DAGNN, S<sup>2</sup>GC, and Grand under ./src/



### DGMLP Training

To test the performance of DGMLP on the Cora, Citeseer, Pubmed dataset, please run this command:

```train
bash ./src/run.sh
```

 

### Node Classification Results:

<img src=".\node_classifi_perf.png" style="zoom:20%;" />
