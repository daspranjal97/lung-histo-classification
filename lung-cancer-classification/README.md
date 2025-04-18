# Lung Cancer Histopathological Image Classification

This repository contains the PyTorch implementation for:
**"Enhanced Lung Cancer Histopathological Image Classification through Dual Transfer Learning and Neighbor Feature Attention-based Pooling"**.

## Dataset
Lung and Colon Cancer Histopathological Image Dataset (LC25000):  
https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4a

Download and extract the dataset, arrange it like this:

```
data/LC25000/
├── adenocarcinoma/
├── squamous_cell_carcinoma/
└── benign/
```

## How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Train the model:
```
python src/train.py --data_path ./data/LC25000
```

3. Evaluate the model:
```
python src/evaluate.py --data_path ./data/LC25000
```

## Authors
- Pranjal Das
- Rajagopal Kumar
- Dushmanta Kumar Das
