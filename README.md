# Autonomous driving sim_agent in waymo competition
Autonomous driving sim_agent in waymo competition

# Prenrender
First we need to prepare data for training. The prerender script will convert the original data format into set of .npz files each containing the data for a single target agent. From code folder run
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