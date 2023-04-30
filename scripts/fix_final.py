import os

import numpy as np
import torch

import mmint_utils
from scripts.diffmpm_torch_batch import SceneBatch, visualize

out_dir = "out/final_test/"

for goal_idx in range(1, 5):
    scene = SceneBatch()
    scene.reset()

    goal_x = np.load("goal_%d.npz" % goal_idx)["goal"]
    goal_x = torch.tensor(goal_x, dtype=scene.dtype, device=scene.device)

    goal_out_dir = os.path.join(out_dir, "goal_%d" % goal_idx)

    final_out_dir = os.path.join(goal_out_dir, "final")
    mmint_utils.make_dir(final_out_dir)

    # Load final data.
    final_data = mmint_utils.load_gzip_pickle(os.path.join(goal_out_dir, "iter_29.pkl"))

    final_x = final_data["x"]
    scene.x = final_x

    sphere_end_pos = torch.tensor(final_data["sphere_end_pos"], dtype=scene.dtype, device=scene.device)
    print(sphere_end_pos)
    scene.init_sphere_tensors(sphere_end_pos)

    visualize(scene, final_out_dir, goal_x=goal_x)
