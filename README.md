# SC-Diff
code for Structure-Conditioned Diffusion for Temporal Knowledge Graph Reasoning
## Requirements
```
dgl==1.1.2
dgl==1.1.2+cu117
fitlog==0.9.15
info_nce_pytorch==0.1.4
numpy==1.24.4
pandas==1.1.3
rdflib==7.0.0
scipy==1.14.0
torch==1.13.1+cu117
torch_scatter==2.0.9
tqdm==4.65.0
transformers==4.20.1
```

## Data preparation

run `src/unseen_event.py` and `src/tri2seq.py`.

## Train and Evaluate
```
python src/main_21.py --dataset ICEWS14

