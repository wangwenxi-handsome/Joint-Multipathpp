import os
import tensorflow as tf
import sys
from tqdm import tqdm
sys.path.append("/root/competition/waymo_sim_agent/prerender")

from features_description import generate_features_description
from configs import get_vectorizer_config
from waymo_open_dataset.protos import scenario_pb2




def data_to_numpy(data):
    for k, v in data.items():
        data[k] = v.numpy()
    data["parsed"] = True


def parse_tfexample_data(filepath):
    vectorizer_config = get_vectorizer_config("N32CloseSegAndValidAgentRenderer")
    vectorizer = vectorizer_config["class"](vectorizer_config)
    dataset = tf.data.TFRecordDataset([filepath])
    sids = []
    for data in tqdm(dataset.as_numpy_iterator()):
        data = tf.io.parse_single_example(data, generate_features_description())
        data_to_numpy(data)
        scene_data = vectorizer.render(data, os.path.basename(filepath))
        scene_id = str(scene_data["scenario_id"])
        sids.append(scene_id)
    return sids


def parse_scenario_data(filepath):
    dataset = tf.data.TFRecordDataset([filepath])
    sids = []
    for scenario_bytes in tqdm(dataset.as_numpy_iterator()):
        scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
        scene_id = str(scenario.scenario_id)
        sids.append(scene_id)
    return sids


if __name__ == "__main__":
    sid1 = parse_tfexample_data("/root/competition/waymo_sim_agent/tools/uncompressed_tf_example_validation_validation_tfexample.tfrecord-00000-of-00150")
    for i in sid1:
        assert os.path.exists(os.path.join("/root/competition/dataset/prerender/validation/", f"scid_{i}.npz"))
