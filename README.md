# Joint Multipath++ for Sim Agent in waymo competition 2023
Autonomous driving sim_agent in waymo competition

# Prenrender
First we need to prepare data for training. The prerender script will convert the original data format into set of .npz files each containing the data for one scene. From code folder run
```
python3 prerender/prerender.py \
    --data_path /path/to/original/data \
    --output_path /output/path/to/prerendered/data \
    --config NCloseSegAndValidAgentRenderer
```
The prerender module is completely self-contained.

# Model
## Encoder
![image](docs/encoder.jpg)

# Train
```
python3 train.py \
    --train_data_path /path/to/train/data \
    --val_data_path /path/to/validation/data \
    --config configs/Multipathpp32.yaml
    --save_folder /save/path
```

# Rollout
```
python3 rollout.py \
    --test_data_path /path/to/test/data \
    --model_path /path/to/model \
    --config configs/Multipathpp32.yaml \
    --save_path /path/to/save/output
```
