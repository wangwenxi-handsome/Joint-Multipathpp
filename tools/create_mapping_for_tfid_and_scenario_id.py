import os
import pickle
import numpy as np
from tqdm import tqdm


source_root = "/root/competition/dataset/prerender/validation/"
dst_path = "./validation_tfid2scneario_id.pkl"
tf_exampleid2scenario_ids = {}
for file in tqdm(os.listdir(source_root)):
    file = os.path.join(source_root, file)
    source_data = dict(np.load(file, allow_pickle=True))
    if "arr_0" in source_data:
        source_data = source_data["arr_0"].item()
    tf_id = str(source_data["file_name"])
    scenario_id = str(source_data["scenario_id"])
    if tf_id not in tf_exampleid2scenario_ids:
        tf_exampleid2scenario_ids[tf_id] = [scenario_id]
    else:
        tf_exampleid2scenario_ids[tf_id].append(scenario_id)
with open(dst_path, "wb") as f:
    pickle.dump(tf_exampleid2scenario_ids, f)
