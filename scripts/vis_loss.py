import matplotlib.pyplot as plt
import os

import mmint_utils

# out_dir = "out/final_test/"
out_dir = "out/in_variation"

for goal_idx in range(4):
    goal_out_dir = os.path.join(out_dir, "goal_{}".format(goal_idx))

    losses = []
    for i in range(30):
        data_i = mmint_utils.load_gzip_pickle(os.path.join(goal_out_dir, "iter_{}.pkl".format(i)))
        losses.append(data_i["loss"])

    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Seed {}".format(goal_idx + 1))
    plt.savefig(os.path.join(goal_out_dir, "loss.png"))
    plt.close()
