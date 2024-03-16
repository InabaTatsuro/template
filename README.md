## Environment
python: 3.10.10

## Setup
```bash
pip install x-transformers numpy omegaconf wandb tqdm
```

## Usage example
### preprocess
```bash
python preprocess.py --out_dir dataset_demo
```

### train
```bash
python train.py --config configs/lr_5e-4.yaml
```

### Generate
```bash
python generate.py --config configs/lr_5e-4.yaml --mode continuation --initial_length 10
```
or
```bash
ython generate.py --config configs/lr_5e-4.yaml --mode scratch --scrach_num 10
```
