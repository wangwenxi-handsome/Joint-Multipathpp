import tensorflow as tf
import numpy as np
from utils.features_description import generate_features_description
from utils.utils import data_to_numpy, get_config
from utils.vectorizer import MultiPathPPRenderer


visualizers_config = get_config("configs/prerender.yaml")[0]
prerender = MultiPathPPRenderer(visualizers_config["renderer_config"])

raw_file = "E:\\motion_dataset\\training\\training_tfexample.tfrecord-00002-of-01000"
dataset = tf.data.TFRecordDataset([raw_file], num_parallel_reads=1)

for data in dataset.as_numpy_iterator():
    data = tf.io.parse_single_example(data, generate_features_description())
    data_to_numpy(data)

    # prerender the scene
    data = prerender.render(data)[0]
    print(data["road_network_embeddings"])
    for i in data:
        if isinstance(data[i], np.ndarray):
            print(i, data[i].shape)
        else:
            print(i, data[i])
    
    # only one scene
    break

"""
# agent order check: in one data, agent is orderd by [target_agents, ego, other_agents]
for data in dataset.as_numpy_iterator():
    data = tf.io.parse_single_example(data, generate_features_description())
    data_to_numpy(data)

    ego_id = np.where(data["state/is_sdc"] == 1)[0][0]
    target_id = list(np.where(data["state/tracks_to_predict"] == 1)[0])
    print(ego_id, target_id)

    assert len(target_id) > 0
    assert target_id[0] == 0
    assert target_id == list(range(ego_id)) or target_id == list(range(ego_id + 1))
"""
