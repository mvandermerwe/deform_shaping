import argparse
import os
import shutil

# parser = argparse.ArgumentParser(description="Imgs to imgs.")
# parser.add_argument("input_dir", type=str, help="Input directory.")
# args = parser.parse_args()

out_dir = "out/in_variation/"
num_goals = 4

video_dir = os.path.join(out_dir, "videos")

for goal_idx in range(num_goals):
    goal_dir = os.path.join(out_dir, "goal_{}".format(goal_idx))
    for iter_dir in ["iter_0", "iter_10", "iter_20", "final"]:
        input_dir = os.path.join(goal_dir, iter_dir)

        images_fns = [f for f in os.listdir(input_dir) if ".png" in f and "image" not in f]
        images_fns = sorted(images_fns, key=lambda x: int(x.split(".")[0]))

        for i, img_fn in enumerate(images_fns):
            shutil.copyfile(os.path.join(input_dir, img_fn), os.path.join(input_dir, "image_{}.png".format(i)))

        # call ffmpeg to generate video from images
        os.system(
            "ffmpeg -framerate 10 -i {}/image_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p {}/{}.mp4".format(input_dir,
                                                                                                           video_dir,
                                                                                                           "goal_%d_%s" % (
                                                                                                           goal_idx,
                                                                                                           iter_dir)))
