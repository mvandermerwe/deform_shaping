import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

import deform_shaping.utils as utils
import mmint_utils

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 1025
steps = 1024
gravity = 0.0
target = [0.8, 0.2]

sphere_radius = 0.05

body_w, body_h = 0.3, 0.1
x_base, y_base = 0.0, 0.019
x_offset = 0.5 - 0.15
y_offset = 0.0

eps = 1e-10

bound = 3
coeff = 0.5
softness = 666.0  # From PlasticineLab
friction = 2.0  # ?


class SceneBatch:

    def __init__(self):
        self.dtype = torch.float32
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0")

        self.create_def_body()
        self.reset()

        self.sphere_start_pos = torch.tensor([0.5, 0.2], dtype=self.dtype, device=self.device)
        self.sphere_end_pos = torch.tensor([0.6, 0.15], dtype=self.dtype, device=self.device)
        self.init_sphere_tensors(self.sphere_end_pos)

    def reset(self):
        self.init_tensors()
        self.x[0] = self.x0
        self.F[0] = torch.eye(dim, dtype=self.dtype, device=self.device).repeat(n_particles, 1, 1)

    def create_def_body(self):
        global n_particles

        # Setup points of deformable body in scene.
        w_count = int(body_w / dx) * 2
        h_count = int(body_h / dx) * 2
        real_dx = body_w / w_count
        real_dy = body_h / h_count
        x = []
        for i in range(w_count):
            for j in range(h_count):
                x.append([
                    x_base + (i + 0.5) * real_dx + x_offset,
                    y_base + (j + 0.5) * real_dy + y_offset
                ])
        self.x0 = torch.tensor(x, dtype=self.dtype, device=self.device)
        n_particles = self.x0.shape[0]

    def init_tensors(self):
        # self.x = torch.zeros([max_steps, n_particles, dim], dtype=self.dtype, device=self.device)
        # self.v = torch.zeros([max_steps, n_particles, dim], dtype=self.dtype, device=self.device)
        # self.C = torch.zeros([max_steps, n_particles, dim, dim], dtype=self.dtype, device=self.device)
        # self.F = torch.zeros([max_steps, n_particles, dim, dim], dtype=self.dtype, device=self.device)
        self.sphere_x = torch.zeros([max_steps, dim], dtype=self.dtype, device=self.device)
        self.sphere_v = torch.zeros([max_steps, dim], dtype=self.dtype, device=self.device)

        self.x = [torch.zeros([n_particles, dim], dtype=self.dtype, device=self.device).requires_grad_(True) for _ in
                  range(max_steps)]
        self.v = [torch.zeros([n_particles, dim], dtype=self.dtype, device=self.device).requires_grad_(True) for _ in
                  range(max_steps)]
        self.C = [torch.zeros([n_particles, dim, dim], dtype=self.dtype, device=self.device).requires_grad_(True) for _
                  in range(max_steps)]
        self.F = [torch.zeros([n_particles, dim, dim], dtype=self.dtype, device=self.device).requires_grad_(True) for _
                  in range(max_steps)]

        self.grid_v_in = torch.zeros([n_grid, n_grid, dim], dtype=self.dtype, device=self.device)
        self.grid_m_in = torch.zeros([n_grid, n_grid], dtype=self.dtype, device=self.device)
        self.grid_v_out = torch.zeros([n_grid, n_grid, dim], dtype=self.dtype, device=self.device)

    def init_sphere_tensors(self, sphere_end_pos: torch.Tensor):
        step_idx = torch.arange(max_steps, dtype=self.dtype, device=self.device)
        self.sphere_x[:, 0] = torch.lerp(self.sphere_start_pos[0], sphere_end_pos[0], step_idx / (max_steps - 1))
        self.sphere_x[:, 1] = torch.lerp(self.sphere_start_pos[1], sphere_end_pos[1], step_idx / (max_steps - 1))
        self.sphere_v[:] = (sphere_end_pos - self.sphere_start_pos) / ((max_steps - 1) * dt)

    def clear_grid(self):
        self.grid_v_in = torch.zeros([n_grid, n_grid, dim], dtype=self.dtype, device=self.device)
        self.grid_m_in = torch.zeros([n_grid, n_grid], dtype=self.dtype, device=self.device)
        self.grid_v_out = torch.zeros([n_grid, n_grid, dim], dtype=self.dtype, device=self.device)

    def p2g(self, step):
        base = (self.x[step] * inv_dx - 0.5).floor().int()
        fx = self.x[step] * inv_dx - base.float()
        w = torch.cat([(0.5 * torch.pow(1.5 - fx, 2)).unsqueeze(1),
                       (0.75 - torch.pow(fx - 1, 2)).unsqueeze(1),
                       (0.5 * torch.pow(fx - 0.5, 2)).unsqueeze(1)], dim=1)

        new_F = (torch.eye(dim, dtype=self.dtype, device=self.device) + dt * self.C[step]) @ self.F[step]
        J = new_F.det()

        self.F[step + 1] = new_F
        r, s = utils.polar_decompose2d(new_F)

        mass = 1
        cauchy = 2 * mu * (new_F - r) @ new_F.transpose(-1, -2)
        cauchy += torch.eye(2, dtype=self.dtype, device=self.device) \
                      .reshape((1, 2, 2)).repeat(n_particles, 1, 1) * (la * J * (J - 1))[:, None, None]

        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * self.C[step]

        contr_idcs = torch.zeros([n_particles, 9, dim], dtype=torch.int64, device=self.device)
        v_in_contr = torch.zeros([n_particles, 9, dim], dtype=self.dtype, device=self.device)
        m_in_contr = torch.zeros([n_particles, 9], dtype=self.dtype, device=self.device)
        for i in range(3):
            for j in range(3):
                offset = torch.tensor([i, j], dtype=self.dtype, device=self.device)
                dpos = (offset - fx) * dx
                weight = w[:, i, 0] * w[:, j, 1]

                contr_idcs[:, i * 3 + j, :] = ((base + offset)[:, 0] * n_grid + (base + offset)[:, 1])[:, None]
                v_in_contr[:, i * 3 + j] = weight[:, None] * (
                        mass * self.v[step] + (affine @ dpos.unsqueeze(-1)).squeeze(-1))
                m_in_contr[:, i * 3 + j] = weight * mass
                # self.grid_v_in[base[:, 0] + i, base[:, 1] + j] += weight[:, None] * (
                #         mass * self.v[step] + (affine @ dpos.unsqueeze(-1)).squeeze(-1))
                # self.grid_m_in[base[:, 0] + i, base[:, 1] + j] += weight * mass
        contr_idcs = contr_idcs.reshape(n_particles * 9, dim)
        v_in_contr = v_in_contr.reshape(n_particles * 9, dim)
        m_in_contr = m_in_contr.reshape(n_particles * 9)
        # Use scatter_add_ to add contributions to self.grid_v_in and self.grid_m_in
        self.grid_v_in.reshape(n_grid ** 2, -1).scatter_add_(dim=0, index=contr_idcs, src=v_in_contr).reshape(n_grid,
                                                                                                              n_grid, 2)
        self.grid_m_in.flatten().scatter_add_(dim=0, index=contr_idcs[:, 0], src=m_in_contr).reshape(n_grid, n_grid)

    def sdf(self, step, grid_pos):
        grid_xy = grid_pos * dx
        sphere_xy = self.sphere_x[step]
        return (torch.norm(grid_xy - sphere_xy, dim=-1) + 1e-8) - sphere_radius

    def normal(self, step, grid_pos):
        grid_xy = grid_pos * dx
        sphere_xy = self.sphere_x[step]
        return (grid_xy - sphere_xy) / (torch.norm(grid_xy - sphere_xy, dim=-1)[:, :, None] + 1e-8)

    def collider_v(self, step, grid_pos, dt_):
        sphere_vel = self.sphere_v[step]

        return sphere_vel  # / dt_

    def collide(self, step, grid_pos, v_out, dt_):
        dist = self.sdf(step, grid_pos)
        influence = torch.min(torch.exp(-dist * softness), torch.tensor(1.0, dtype=self.dtype, device=self.device))

        mask = torch.logical_or(dist <= 0, influence > 0.1)

        D = self.normal(step, grid_pos)
        collider_v_at_grid = self.collider_v(step, grid_pos, dt_)

        input_v = v_out - collider_v_at_grid
        normal_component = torch.einsum("ijk,ijk->ij", input_v, D)

        grid_v_t = input_v - (torch.min(normal_component, torch.tensor(0.0, dtype=self.dtype, device=self.device))[:, :,
                              None] * D)
        grid_v_t_norm = torch.norm(grid_v_t, dim=-1) + 1e-8
        norm_mask = grid_v_t_norm > 1e-30

        grid_v_t_friction = grid_v_t / grid_v_t_norm[:, :, None] * torch.max(
            torch.tensor(0, dtype=self.dtype, device=self.device), grid_v_t_norm + normal_component * friction)[:, :,
                                                                   None]
        flag = (normal_component < 0).float()
        grid_v_t = (grid_v_t * (1.0 - norm_mask[:, :, None].float())) + (
                (grid_v_t_friction * flag[:, :, None] + grid_v_t * (1 - flag)[:, :, None]) * norm_mask[:, :,
                                                                                             None].float())

        v_out_new = (collider_v_at_grid + input_v * (1 - influence)[:, :, None] + grid_v_t * influence[:, :,
                                                                                             None]) * mask.float()[:, :,
                                                                                                      None] + v_out * (
                                                                                                                              1.0 - mask.float())[
                                                                                                                      :,
                                                                                                                      :,
                                                                                                                      None]
        return v_out_new

    def grid_op(self, step):
        # Create grid of indices.
        x_s = torch.arange(0, n_grid, dtype=self.dtype, device=self.device)
        y_s = torch.arange(0, n_grid, dtype=self.dtype, device=self.device)
        x, y = torch.meshgrid(x_s, y_s)
        grid_pos = torch.stack([x, y], dim=2)

        inv_m = 1 / (self.grid_m_in + eps)
        v_out = inv_m[:, :, None] * self.grid_v_in

        v_out_2 = self.collide(step, grid_pos, v_out, dt)

        zero_mask = torch.logical_and(grid_pos[:, :, 0] < bound, v_out_2[:, :, 0] < 0)
        zero_mask = torch.logical_or(zero_mask,
                                     torch.logical_and(grid_pos[:, :, 0] > n_grid - bound, v_out_2[:, :, 0] > 0))
        zero_mask = torch.logical_or(zero_mask,
                                     torch.logical_and(grid_pos[:, :, 1] > n_grid - bound, v_out_2[:, :, 1] > 0))
        v_out_3 = v_out_2 * (1.0 - zero_mask.float())[:, :, None]

        # Friction.
        friction_mask = torch.logical_and(grid_pos[:, :, 1] < bound, v_out_3[:, :, 1] < 0)
        normal = torch.tensor([0, 1], dtype=self.dtype, device=self.device).unsqueeze(0).unsqueeze(0).repeat(n_grid,
                                                                                                             n_grid, 1)
        lin = torch.einsum("ijk,ijk->ij", v_out_3, normal)
        lin_mask = lin < 0

        vit = v_out_3 - lin[:, :, None] * normal
        lit = torch.norm(vit, dim=-1) + 1e-10

        lit_mask = lit + coeff * lin <= 0

        zero_lit_mask = torch.logical_and(friction_mask, torch.logical_and(lin_mask, lit_mask))
        v_out_4 = v_out_3 * (1.0 - zero_lit_mask.float())[:, :, None]

        non_zero_lit_mask = torch.logical_and(friction_mask,
                                              torch.logical_and(lin_mask, torch.logical_not(lit_mask)))
        v_out_5 = ((1 + coeff * lin / lit)[:, :, None] * vit) * non_zero_lit_mask.float()[:, :, None] + \
                  v_out_4 * (1.0 - non_zero_lit_mask.float())[:, :, None]

        self.grid_v_out = v_out_5

    def g2p(self, step):
        base = (self.x[step] * inv_dx - 0.5).floor().int()
        fx = self.x[step] * inv_dx - base.float()
        w = torch.cat([(0.5 * torch.pow(1.5 - fx, 2)).unsqueeze(1),
                       (0.75 - torch.pow(fx - 1, 2)).unsqueeze(1),
                       (0.5 * torch.pow(fx - 0.5, 2)).unsqueeze(1)], dim=1)

        new_v = torch.zeros([n_particles, 9, 2], dtype=self.dtype, device=self.device)
        new_C = torch.zeros([n_particles, 9, 2, 2], dtype=self.dtype, device=self.device)

        for i in range(3):
            for j in range(3):
                dpos = torch.tensor([i, j], dtype=self.dtype, device=self.device) - fx
                g_v = self.grid_v_out[base[:, 0] + i, base[:, 1] + j]
                weight = w[:, i, 0] * w[:, j, 1]

                new_v[:, 3 * i + j] = weight[:, None] * g_v
                new_C[:, 3 * i + j] = 4 * weight[:, None, None] * torch.einsum("bi,bj->bij", g_v, dpos) * inv_dx

        self.v[step + 1] = new_v.sum(dim=1)
        self.x[step + 1] = self.x[step] + dt * self.v[step + 1]
        self.C[step + 1] = new_C.sum(dim=1)

    def advance(self, step):
        self.clear_grid()
        self.p2g(step)
        self.grid_op(step)
        self.g2p(step)


def visualize(scene_: SceneBatch, out_dir: str = None, goal_x: torch.Tensor = None):
    if out_dir is not None:
        mmint_utils.make_dir(out_dir)
    plt.ion()
    fig, ax = plt.subplots()

    for step in range(15, max_steps, 16):
        # for step in range(max_steps):
        ax.clear()
        ax.axis("off")
        plt.xlim(0.2, 0.8)
        plt.ylim(0.0, 0.6)
        circle = plt.Circle(
            (scene_.sphere_x.detach().cpu().numpy()[step][0], scene_.sphere_x.detach().cpu().numpy()[step][1]),
            sphere_radius, color='r')
        ax.add_patch(circle)
        ax.plot([0.05, 0.95], [0.02, 0.02], color='black', linewidth=2)
        ax.scatter(scene_.x[step].detach().cpu().numpy()[:, 0], scene_.x[step].detach().cpu().numpy()[:, 1], c='b',
                   s=0.1)
        if goal_x is not None:
            ax.scatter(goal_x.cpu().numpy()[:, 0], goal_x.cpu().numpy()[:, 1], c='g', alpha=0.5, s=0.1)
        ax.set_aspect("equal")
        if out_dir is not None:
            plt.savefig(os.path.join(out_dir, "{}.png".format(step)), dpi=150)
        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.close()
        # plt.show()
        plt.pause(0.1)

    plt.close()


def loss(scene_: SceneBatch, goal_x_: torch.Tensor):
    loss_ = (scene_.x[-1] - goal_x_).norm(dim=-1).mean()
    return loss_


if __name__ == '__main__':
    # initialization
    scene = SceneBatch()

    if False:
        goal_x = np.load("goal_1.npz")["goal"]
        goal_x = torch.tensor(goal_x, dtype=scene.dtype, device=scene.device)

        sphere_end_pos = torch.tensor([0.5, 0.15], dtype=scene.dtype, device=scene.device).requires_grad_(True)
        opt = torch.optim.Adam([sphere_end_pos], lr=1e-2)

        start = time.time()
        for i in trange(30):
            scene.reset()
            scene.init_sphere_tensors(sphere_end_pos)

            for s in range(max_steps - 1):
                scene.advance(s)

            l = loss(scene, goal_x)

            opt.zero_grad()
            l.backward()
            opt.step()

            print("step: {}, loss: {}, end_pos: {}".format(i, l.item(), sphere_end_pos.detach().cpu().numpy()))

            if i % 10 == 0:
                visualize(scene, goal_x=goal_x)
        visualize(scene, goal_x=goal_x)
    elif False:
        out_dir = "out/in_variation/"
        mmint_utils.make_dir(out_dir)

        goal_x = np.load("goal_1.npz")["goal"]
        goal_x = torch.tensor(goal_x, dtype=scene.dtype, device=scene.device)

        init_goals = [[0.5, 0.15], [0.4, 0.15], [0.6, 0.17], [0.5, 0.2]]

        for init_goal_idx in range(3, len(init_goals)):
            init_goal = init_goals[init_goal_idx]
            sphere_end_pos = torch.tensor(init_goal, dtype=scene.dtype, device=scene.device).requires_grad_(True)

            goal_out_dir = os.path.join(out_dir, "goal_%d" % init_goal_idx)
            mmint_utils.make_dir(goal_out_dir)

            opt = torch.optim.Adam([sphere_end_pos], lr=1e-2)

            for i in trange(30):
                scene.reset()
                scene.init_sphere_tensors(sphere_end_pos)

                for s in range(max_steps - 1):
                    scene.advance(s)

                l = loss(scene, goal_x)

                opt.zero_grad()
                l.backward()
                opt.step()

                mmint_utils.save_gzip_pickle({
                    "loss": l.item(),
                    "step": i,
                    "sphere_end_pos": sphere_end_pos.detach().cpu().numpy(),
                    "x": scene.x,
                }, os.path.join(goal_out_dir, "iter_%d.pkl" % i))

                print("step: {}, loss: {}, end_pos: {}".format(i, l.item(), sphere_end_pos.detach().cpu().numpy()))

                if i % 10 == 0:
                    visualize(scene, os.path.join(goal_out_dir, "iter_%d" % i), goal_x)
            visualize(scene, os.path.join(goal_out_dir, "final"), goal_x)
    elif True:
        out_dir = "out/final_test_fix/"
        mmint_utils.make_dir(out_dir)

        for goal_idx in range(1, 5):
            goal_x = np.load("goal_%d.npz" % goal_idx)["goal"]
            goal_x = torch.tensor(goal_x, dtype=scene.dtype, device=scene.device)

            goal_out_dir = os.path.join(out_dir, "goal_%d" % goal_idx)
            mmint_utils.make_dir(goal_out_dir)

            sphere_end_pos = torch.tensor([0.5, 0.15], dtype=scene.dtype, device=scene.device).requires_grad_(True)
            opt = torch.optim.Adam([sphere_end_pos], lr=1e-2)

            start = time.time()
            for i in trange(30):
                scene.reset()
                scene.init_sphere_tensors(sphere_end_pos)

                for s in range(max_steps - 1):
                    scene.advance(s)

                l = loss(scene, goal_x)

                opt.zero_grad()
                l.backward()
                opt.step()

                mmint_utils.save_gzip_pickle({
                    "loss": l.item(),
                    "step": i,
                    "sphere_end_pos": sphere_end_pos.detach().cpu().numpy(),
                    "x": scene.x,
                }, os.path.join(goal_out_dir, "iter_%d.pkl" % i))

                print("step: {}, loss: {}, end_pos: {}".format(i, l.item(), sphere_end_pos.detach().cpu().numpy()))

                if i % 10 == 0:
                    visualize(scene, os.path.join(goal_out_dir, "iter_%d" % i), goal_x)
            visualize(scene, os.path.join(goal_out_dir, "final"), goal_x)
            end = time.time()
            print("goal %d: %f" % (goal_idx, end - start))
    elif False:
        goal_poses = [
            [0.5, 0.15],
            [0.6, 0.15],
            [0.4, 0.15],
            [0.45, 0.13],
            [0.55, 0.16],
        ]

        out_dir = "out/final_goals/"
        mmint_utils.make_dir(out_dir)

        for goal_idx in range(len(goal_poses)):
            goal_out_dir = os.path.join(out_dir, "goal_%d" % goal_idx)
            mmint_utils.make_dir(goal_out_dir)

            sphere_end_pos = torch.tensor(goal_poses[goal_idx], dtype=scene.dtype, device=scene.device)
            scene.reset()
            scene.init_sphere_tensors(sphere_end_pos)

            for s in trange(max_steps - 1):
                scene.advance(s)

            np.savez("goal_%d.npz" % goal_idx, goal=scene.x[-1].detach().cpu().numpy())

            visualize(scene, goal_out_dir)
    else:
        sphere_end_pos = torch.tensor([0.5, 0.15], dtype=scene.dtype, device=scene.device)
        scene.reset()
        scene.init_sphere_tensors(sphere_end_pos)

        for s in trange(max_steps - 1):
            scene.advance(s)

        mmint_utils.save_gzip_pickle({
            "x": scene.x,
            "v": scene.v,
            "C": scene.C,
            "F": scene.F,
            "sphere_x": scene.sphere_x,
            "sphere_v": scene.sphere_v,
        }, "scene.pkl.gzip")

        visualize(scene)
