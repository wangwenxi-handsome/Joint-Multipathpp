import os
import time
import numpy as np
import tensorflow as tf
from features_description import generate_features_description
from configs import get_vectorizer_config
from vectorizer import SegmentAndAgentSequenceRender
from prerender import data_to_numpy


def prerender_one_scene(file_path):
    vectorizer_config = get_vectorizer_config("N16CloseSegAndValidAgentRenderer")
    vectorizer = SegmentAndAgentSequenceRender(vectorizer_config)
    dataset = tf.data.TFRecordDataset([file_path], num_parallel_reads=1)
    for data in dataset.as_numpy_iterator():
        data = tf.io.parse_single_example(data, generate_features_description())
        data_to_numpy(data)

        # prerender the scene
        data = vectorizer.render(data, os.path.basename(file_path))
        print(data[""])
        for i in data:
            if isinstance(data[i], np.ndarray):
                print(i, data[i].shape)
            else:
                print(i, data[i])

        # only one scene
        # there are many scenes in one tfrecords file
        break


# agent_order_check: check if in one data, agent is orderd by [target_agents, ego, other_agents]
def agent_order_check(file_path):
    dataset = tf.data.TFRecordDataset([file_path], num_parallel_reads=1)
    for data in dataset.as_numpy_iterator():
        data = tf.io.parse_single_example(data, generate_features_description())
        data_to_numpy(data)

        ego_id = np.where(data["state/is_sdc"] == 1)[0][0]
        target_id = list(np.where(data["state/tracks_to_predict"] == 1)[0])
        print(ego_id, target_id)

        assert len(target_id) > 0
        assert target_id[0] == 0
        assert target_id == list(range(ego_id)) or target_id == list(range(ego_id + 1))


if __name__ == "__main__":
    file_path = "E:\\motion_dataset\\training\\training_tfexample.tfrecord-00001-of-01000"
    # agent_order_check(file_path)
    prerender_one_scene(file_path)