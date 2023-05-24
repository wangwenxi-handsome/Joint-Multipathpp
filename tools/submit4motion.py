"""
Submit for motion prediction.
Npz files maked by unzip_tf_record.py (line 124) and raw tf sample data (line 27) are needed.
"""

import os
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import motion_submission_pb2
import sys
sys.path.append("../prerender")

from features_description import generate_features_description
from configs import get_vectorizer_config

OUTPUT_ROOT_DIRECTORY = './result6'

def main():
    os.makedirs(OUTPUT_ROOT_DIRECTORY, exist_ok=True)
    output_filenames = []

    VALIDATION_FILES = "/root/competition/dataset/tfsample/testing/*tfrecord*"  # tf sample data
    filenames = tf.io.matching_files(VALIDATION_FILES)
    vectorizer_config = get_vectorizer_config("N32CloseSegAndValidAgentRenderer")
    vectorizer = vectorizer_config["class"](vectorizer_config)
    # Iterate over shards. This could be parallelized in any custom way, as the
    # number of output shards is not required to be the same as the initial dataset.
    for shard_filename in tqdm(filenames):
    # A shard filename has the structure: `validation.tfrecord-00000-of-00150`.
    # We want to maintain the same shard naming here, for simplicity, so we can
    # extract the suffix.
        dataset = tf.data.TFRecordDataset([shard_filename])
        scenario_predictions = []
        shard_suffix = shard_filename.numpy().decode('utf8')[-len('-00000-of-00150'):]    
        for data in dataset.as_numpy_iterator():
            data = tf.io.parse_single_example(data, generate_features_description())
            data_to_numpy(data)
            scene_id = str(data["scenario/id"])[3:-2]
            scenario_predictions.append(scenario_rollouts_from_states(scene_id, data["state/tracks_to_predict"]))
        

        # Now that we have 2 `ScenarioRollouts` for this shard, we can package them
        # into a `SimAgentsChallengeSubmission`. Remember to populate the metadata
        # for each shard.
        shard_submission = motion_submission_pb2.MotionChallengeSubmission(
            account_name='zhenghaotian1998@163.com',
            unique_method_name='multipath',
            authors=['WenxiWang', 'Haotian Zhen'],
            affiliation='waymo',
            description='Submission from the motion prediction',
            method_link='https://waymo.com/open/',
            submission_type=motion_submission_pb2.MotionChallengeSubmission.MOTION_PREDICTION,
            uses_lidar_data=False,
            scenario_predictions=scenario_predictions
        )

        # Now we can export this message to a binproto, saved to local storage.
        output_filename = f'submission.binproto{shard_suffix}'
        with open(os.path.join(OUTPUT_ROOT_DIRECTORY, output_filename), 'wb') as f:
            f.write(shard_submission.SerializeToString())
        output_filenames.append(output_filename)

    # Once we have created all the shards, we can package them directly into a
    # tar.gz archive, ready for submission.
    with tarfile.open(
        os.path.join(OUTPUT_ROOT_DIRECTORY, 'submission.tar.gz'), 'w:gz') as tar:
        for output_filename in output_filenames:
            tar.add(os.path.join(OUTPUT_ROOT_DIRECTORY, output_filename),
                    arcname=output_filename)


def data_to_numpy(data):
    for k, v in data.items():
        data[k] = v.numpy()
    data["parsed"] = True


def one_predictions_from_states(
    pred_states: np.ndarray, confidence: np.ndarray, agent_id: int
    ) -> motion_submission_pb2.SingleObjectPrediction:
  # States shape: (num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  trajectories = []
  for i_rollout in range(len(pred_states)):
    trajectory = motion_submission_pb2.Trajectory(center_x=pred_states[i_rollout, :, 0], center_y=pred_states[i_rollout, :, 1])
    trajectories.append(motion_submission_pb2.ScoredTrajectory(
        trajectory = trajectory, confidence = confidence[i_rollout]
    ))

  return motion_submission_pb2.SingleObjectPrediction(
    object_id = agent_id, trajectories=trajectories)
  

def scenario_rollouts_from_states(
    scenario_id: str, tracks_to_predict: np.ndarray
    ) -> motion_submission_pb2.ChallengeScenarioPredictions:
  # States shape: (num_rollouts, num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  single_predictions = []
  pred_data = parse_one_pred_file(scenario_id)
  pred_states = pred_data["states"]
  pred_states = pred_states[:, :, ::5, :]
  agent_ids = pred_data["agent_id"]
  confidence = np.ones((pred_states.shape[0], pred_states.shape[1]), dtype=np.float32)
#   pred_data["confidence"]
  idxs = np.argwhere(tracks_to_predict==1)

  for id in idxs:
    id = id[0]
    single_predictions.append(one_predictions_from_states(pred_states[:, id, ...], confidence[:, id, ...], agent_ids[id]))

  single_predictions = motion_submission_pb2.PredictionSet(predictions = single_predictions)
  return motion_submission_pb2.ChallengeScenarioPredictions(
      # Note: remember to include the Scenario ID in the proto message.
      single_predictions=single_predictions, scenario_id=scenario_id)


def parse_one_pred_file(scenario_id: str):
    file = os.path.join("/root/competition/dataset/output/testing6_decompress/", f"scid_{scenario_id}.npz")
    source_data = np.load(file)
    return source_data


if __name__ == "__main__":
    main()
