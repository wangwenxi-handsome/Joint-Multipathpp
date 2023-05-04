import os
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


from waymo_open_dataset.protos import sim_agents_submission_pb2


SOURCE_DATA_ROOT = "/root/competition/dataset/prerender/validation/"
PRED_ROOT = "/root/competition/dataset/prerender/validation/"
TF_EXAMPLE_ID_2_SCENARIO_ID_PATH = "/root/competition/waymo_sim_agent/tools/validation_tfid2scneario_id.pkl"
# Where results are going to be saved.
OUTPUT_ROOT_DIRECTORY = './results'





def main():
    os.makedirs(OUTPUT_ROOT_DIRECTORY, exist_ok=True)
    output_filenames = []

    with open(TF_EXAMPLE_ID_2_SCENARIO_ID_PATH, "rb") as f:
        tf_exampleid2scenario_ids = pickle.load(f)
    # Iterate over shards. This could be parallelized in any custom way, as the
    # number of output shards is not required to be the same as the initial dataset.
    for shard_filename in tqdm(tf_exampleid2scenario_ids.keys()):
        # A shard filename has the structure: `validation.tfrecord-00000-of-00150`.
        # We want to maintain the same shard naming here, for simplicity, so we can
        # extract the suffix.
        shard_suffix = shard_filename[-15:]
        scenario_rollouts = []
        for scenario_id in tf_exampleid2scenario_ids[shard_filename]:
            source_data = dict(np.load(os.path.join(SOURCE_DATA_ROOT, "scid_"+scenario_id+".npz"), allow_pickle=True))
            if "arr_0" in source_data:
                source_data = source_data["arr_0"].item()
            simulated_states = get_pred_states(scenario_id)
            object_id = source_data["agent_id"].astype(np.int32)
            # object_id = object_id[:source_data["target_id"]+1]
            object_id = object_id[object_id!=-1]
            assert simulated_states.shape[1] == len(object_id)
            sr = scenario_rollouts_from_states(
                scenario_id, simulated_states, object_id)
            scenario_rollouts.append(sr)

        # Now that we have 2 `ScenarioRollouts` for this shard, we can package them
        # into a `SimAgentsChallengeSubmission`. Remember to populate the metadata
        # for each shard.
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=scenario_rollouts,
            submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
            account_name='zhenghaotian1998@163.com',
            unique_method_name='sim_agents_tutorial',
            authors=['test'],
            affiliation='waymo',
            description='Submission from the Sim Agents tutorial',
            method_link='https://waymo.com/open/'
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


def joint_scene_from_states(
    states: np.ndarray, object_ids: np.ndarray
) -> sim_agents_submission_pb2.JointScene:
    # States shape: (num_objects, num_steps, 4).
    # Objects IDs shape: (num_objects,).
    simulated_trajectories = []
    for i_object in range(len(object_ids)):
      simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
          center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
          center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
          object_id=object_ids[i_object]
      ))
    return sim_agents_submission_pb2.JointScene(
        simulated_trajectories=simulated_trajectories)


def scenario_rollouts_from_states(
    scenario_id: str,
    states: np.ndarray,
    object_ids
) -> sim_agents_submission_pb2.ScenarioRollouts:
  # States shape: (num_rollouts, num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  joint_scenes = []
  for i_rollout in range(states.shape[0]):
     joint_scenes.append(joint_scene_from_states(states[i_rollout], object_ids))
  return sim_agents_submission_pb2.ScenarioRollouts(
      # Note: remember to include the Scenario ID in the proto message.
      joint_scenes=joint_scenes, scenario_id=scenario_id)


def get_pred_states(scenario_id: int):
    # simulated_states = np.load(os.path.join(PRED_ROOT, "scid_"+scenario_id+".npy"))
    source_data = dict(np.load(os.path.join(PRED_ROOT, "scid_"+str(scenario_id)+".npz"), allow_pickle=True))
    if "arr_0" in source_data:
        source_data = source_data["arr_0"].item()
    xy, heading = source_data["future/xy"], source_data["future/yaw"]
    z = np.zeros(shape=(xy.shape[0], xy.shape[1], 1), dtype=xy.dtype)
    pred = np.concatenate((xy, z, heading), -1)
    pred = pred[source_data["agent_id"]!=-1]
    pred = np.tile(pred[np.newaxis, ...], (32, 1, 1, 1))
    return pred


if __name__ == "__main__":
    main()
