"""
Submit for motion prediction.
Npz files maked by unzip_tf_record.py (line 224) and raw scenario data (line 39) are needed.
"""

import os
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2

from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.utils import trajectory_utils


# Where results are going to be saved.
OUTPUT_ROOT_DIRECTORY = './result32'

DEFAULT_COUNT = 0



def main():
    os.makedirs(OUTPUT_ROOT_DIRECTORY, exist_ok=True)
    output_filenames = []

    VALIDATION_FILES = "/root/competition/dataset/scenario/testing/*tfrecord*"
    filenames = tf.io.matching_files(VALIDATION_FILES)
    # Iterate over shards. This could be parallelized in any custom way, as the
    # number of output shards is not required to be the same as the initial dataset.
    for shard_filename in tqdm(filenames):
    # A shard filename has the structure: `validation.tfrecord-00000-of-00150`.
    # We want to maintain the same shard naming here, for simplicity, so we can
    # extract the suffix.
        shard_suffix = shard_filename.numpy().decode('utf8')[-len('-00000-of-00150'):]
    
        # Now we can iterate over the Scenarios in the shard. To make this faster as
        # part of the tutorial, we will only process 2 Scenarios per shard. Obviously,
        # to create a valid submission, all the scenarios needs to be present.
        shard_dataset = tf.data.TFRecordDataset([shard_filename])
        shard_iterator = shard_dataset.as_numpy_iterator()

        scenario_rollouts = []
        for scenario_bytes in tqdm(shard_iterator):
            scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
            logged_trajectories, simulated_states = simulate_with_extrapolation(
                scenario)
            # print(logged_trajectories.object_id)
            # print(simulated_states.shape)
            sr = scenario_rollouts_from_states(
                scenario, simulated_states, logged_trajectories.object_id)
            submission_specs.validate_scenario_rollouts(sr, scenario)
            scenario_rollouts.append(sr)
        

        # Now that we have 2 `ScenarioRollouts` for this shard, we can package them
        # into a `SimAgentsChallengeSubmission`. Remember to populate the metadata
        # for each shard.
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=scenario_rollouts,
            submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
            account_name='xxx@xxx.com',
            unique_method_name='multipath',
            authors=['xxx'],
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


def simulate_with_extrapolation(
    scenario: scenario_pb2.Scenario,
    print_verbose_comments: bool = False) -> tf.Tensor:
  vprint = print if print_verbose_comments else lambda arg: None

  # To load the data, we create a simple tensorized version of the object tracks.
  logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
  # Using `ObjectTrajectories` we can select just the objects that we need to
  # simulate and remove the "future" part of the Scenario.
  vprint(f'Original shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')
  logged_trajectories = logged_trajectories.gather_objects_by_id(
      tf.convert_to_tensor(submission_specs.get_sim_agent_ids(scenario)))
  logged_trajectories = logged_trajectories.slice_time(
      start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1)
  vprint(f'Modified shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')

  # We can verify that all of these objects are valid at the last step.
  vprint(f'Are all agents valid: {tf.reduce_all(logged_trajectories.valid[:, -1]).numpy()}')

  # We extract the speed of the sim agents (in the x/y/z components) ready for
  # extrapolation (this will be our policy).
  states = tf.stack([logged_trajectories.x, logged_trajectories.y,
                    logged_trajectories.z, logged_trajectories.heading],
                    axis=-1)
  n_objects, n_steps, _ = states.shape
  last_velocities = states[:, -1, :3] - states[:, -2, :3]
  # We also make the heading constant, so concatenate 0. as angular speed.
  last_velocities = tf.concat(
      [last_velocities, tf.zeros((n_objects, 1))], axis=-1)
  # It can happen that the second to last state of these sim agents might be
  # invalid, so we will set a zero speed for them.
  vprint(f'Is any 2nd to last state invalid: {tf.reduce_any(tf.logical_not(logged_trajectories.valid[:, -2]))}')
  vprint(f'This will result in either min or max speed to be really large: {tf.reduce_max(tf.abs(last_velocities))}')
  valid_diff = tf.logical_and(logged_trajectories.valid[:, -1],
                              logged_trajectories.valid[:, -2])
  # `last_velocities` shape: (n_objects, 4).
  last_velocities = tf.where(valid_diff[:, tf.newaxis],
                            last_velocities,
                            tf.zeros_like(last_velocities))
  vprint(f'Now this should be back to a normal value: {tf.reduce_max(tf.abs(last_velocities))}')

  # Now we carry over a simulation. As we discussed, we actually want 32 parallel
  # simulations, so we make this batched from the very beginning. We add some
  # random noise on top of our actions to make sure the behaviours are different.
  # To properly scale the noise, we get the max velocities (average over all
  # objects, corresponding to axis 0) in each of the dimensions (x/y/z/heading).
  NOISE_SCALE = 0.01
  # `max_action` shape: (4,).
  max_action = tf.reduce_max(last_velocities, axis=0)
  # We create `simulated_states` with shape (n_rollouts, n_objects, n_steps, 4).
  simulated_states = tf.tile(states[tf.newaxis, :, -1:, :], [submission_specs.N_ROLLOUTS, 1, 1, 1])
  vprint(f'Shape: {simulated_states.shape}')

  for step in range(submission_specs.N_SIMULATION_STEPS):
    current_state = simulated_states[:, :, -1, :]
    # Random actions, take a normal and normalize by min/max actions
    action_noise = tf.random.normal(
        current_state.shape, mean=0.0, stddev=NOISE_SCALE)
    actions_with_noise = last_velocities[None, :, :] + (action_noise * max_action)
    next_state = current_state + actions_with_noise
    simulated_states = tf.concat(
        [simulated_states, next_state[:, :, None, :]], axis=2)

  # We also need to remove the first time step from `simulated_states` (it was
  # still history).
  # `simulated_states` shape before: (n_rollouts, n_objects, 81, 4).
  # `simulated_states` shape after: (n_rollouts, n_objects, 80, 4).
  simulated_states = simulated_states[:, :, 1:, :]
  vprint(f'Final simulated states shape: {simulated_states.shape}')

  return logged_trajectories, simulated_states


def joint_scene_from_states(
    states: tf.Tensor, object_ids: tf.Tensor,
    pred_states: np.ndarray, pred_agent_id: np.ndarray
    ) -> sim_agents_submission_pb2.JointScene:
  # States shape: (num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  global DEFAULT_COUNT
  states = states.numpy()
  simulated_trajectories = []
  object_ids_npy = object_ids.numpy()
  for i_object in range(len(object_ids)):
    obj_id = np.argwhere(pred_agent_id==object_ids_npy[i_object])
    if len(obj_id) == 1:
        obj_id = obj_id[0, 0]
        simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
            center_x=pred_states[obj_id, :, 0], center_y=pred_states[obj_id, :, 1],
            center_z=pred_states[obj_id, :, 2], heading=pred_states[obj_id, :, 3],
            object_id=object_ids[i_object]
        ))
    else:
        DEFAULT_COUNT += 1
        simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
            center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
            center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
            object_id=object_ids[i_object]
        ))
  return sim_agents_submission_pb2.JointScene(
      simulated_trajectories=simulated_trajectories)
  

def scenario_rollouts_from_states(
    scenario: scenario_pb2.Scenario,
    states: tf.Tensor, object_ids: tf.Tensor
    ) -> sim_agents_submission_pb2.ScenarioRollouts:
  # States shape: (num_rollouts, num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  global DEFAULT_COUNT
  joint_scenes = []
  pred_data = parse_one_pred_file(scenario.scenario_id)
  pred_states = pred_data["states"]
  pred_agent_id = pred_data["agent_id"]
  pre_default_count = DEFAULT_COUNT

  for i_rollout in range(states.shape[0]):
    joint_scenes.append(joint_scene_from_states(states[i_rollout], object_ids, pred_states[i_rollout], pred_agent_id))

  if DEFAULT_COUNT != pre_default_count:
     DEFAULT_COUNT = pre_default_count + 1
  return sim_agents_submission_pb2.ScenarioRollouts(
      # Note: remember to include the Scenario ID in the proto message.
      joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)


def parse_one_pred_file(scenario_id: str):
    file = os.path.join("/root/competition/dataset/output/testing32_decompress/", f"scid_{scenario_id}.npz")
    source_data = np.load(file)
    return source_data


if __name__ == "__main__":
    main()
