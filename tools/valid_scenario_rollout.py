import os
import tensorflow as tf
import tqdm
import numpy as np

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs

from submit import get_pred_states, scenario_rollouts_from_states


SOURCE_DATA_ROOT = "/root/competition/dataset/prerender/validation/"
filenames = ["/root/competition/waymo_sim_agent/tools/uncompressed_scenario_validation_validation.tfrecord-00000-of-00150"]
for shard_filename in filenames:
    # A shard filename has the structure: `validation.tfrecord-00000-of-00150`.
    # We want to maintain the same shard naming here, for simplicity, so we can
    # extract the suffix.
    print(shard_filename)
    shard_suffix = shard_filename[-15:]
    
    # Now we can iterate over the Scenarios in the shard. To make this faster as
    # part of the tutorial, we will only process 2 Scenarios per shard. Obviously,
    # to create a valid submission, all the scenarios needs to be present.
    shard_dataset = tf.data.TFRecordDataset([shard_filename])
    shard_iterator = shard_dataset.as_numpy_iterator()

    scenario_rollouts = []
    for scenario_bytes in tqdm.tqdm(shard_iterator):
        scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
        print("scene id:", scenario.scenario_id)
        source_data = dict(np.load(os.path.join(SOURCE_DATA_ROOT, "scid_"+scenario.scenario_id+".npz"), allow_pickle=True))
        if "arr_0" in source_data:
            source_data = source_data["arr_0"].item()
        simulated_states = get_pred_states(scenario.scenario_id)
        object_id = source_data["agent_id"].astype(np.int32)
        object_id = object_id[object_id!=-1]
        assert simulated_states.shape[1] == len(object_id)
        sr = scenario_rollouts_from_states(
            scenario.scenario_id, simulated_states, object_id)
        scenario_rollouts.append(sr)
        # print(logged_trajectories.object_id)
        # print(simulated_states.shape)
        
        submission_specs.validate_scenario_rollouts(sr, scenario)