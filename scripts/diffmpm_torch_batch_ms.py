import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

import deform_shaping.utils as utils
import mmint_utils
from scripts.diffmpm_torch_batch import SceneBatch

dim = 2
n_particles = 1824
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
max_steps = 1024
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


class SceneBatchMultiShooting:

    def __init__(self):
        self.dtype = torch.float32
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0")
        self.N = 16
        self.steps_per_shot = (max_steps // self.N) + 1

        self.reset()

    def reset(self):
        self.init_tensors()

    def init_tensors(self):
        steps_per_shot = (max_steps // self.N) + 1
        self.sphere_x = torch.zeros([self.N, steps_per_shot, dim], dtype=self.dtype, device=self.device)
        self.sphere_v = torch.zeros([self.N, steps_per_shot, dim], dtype=self.dtype, device=self.device)

        self.x = [torch.zeros([self.N, n_particles, dim], dtype=self.dtype, device=self.device).requires_grad_(True)
                  for _ in range(steps_per_shot)]
        self.v = [torch.zeros([self.N, n_particles, dim], dtype=self.dtype, device=self.device).requires_grad_(True)
                  for _ in range(steps_per_shot)]
        self.C = [torch.zeros([self.N, n_particles, dim, dim], dtype=self.dtype, device=self.device).requires_grad_(
            True) for _ in range(steps_per_shot)]
        self.F = [torch.zeros([self.N, n_particles, dim, dim], dtype=self.dtype, device=self.device).requires_grad_(
            True) for _ in range(steps_per_shot)]

        self.grid_v_in = torch.zeros([self.N, n_grid, n_grid, dim], dtype=self.dtype, device=self.device)
        self.grid_m_in = torch.zeros([self.N, n_grid, n_grid], dtype=self.dtype, device=self.device)
        self.grid_v_out = torch.zeros([self.N, n_grid, n_grid, dim], dtype=self.dtype, device=self.device)

    def init_sphere_tensors(self, x_init, v_init, C_init, F_init, sphere_x_init):
        self.x[0] = x_init
        self.v[0] = v_init
        self.C[0] = C_init
        self.F[0] = F_init

        steps_per_shot = (max_steps // self.N) + 1
        step_idx = torch.arange(steps_per_shot, dtype=self.dtype, device=self.device)
        for idx in range(self.N):
            self.sphere_x[idx, :, 0] = torch.lerp(sphere_x_init[idx, 0], sphere_x_init[idx + 1, 0],
                                                  step_idx / (steps_per_shot - 1))
            self.sphere_x[idx, :, 1] = torch.lerp(sphere_x_init[idx, 1], sphere_x_init[idx + 1, 1],
                                                  step_idx / (steps_per_shot - 1))
            self.sphere_v[idx, :] = (sphere_x_init[idx + 1, :] - sphere_x_init[idx, :]) / ((steps_per_shot - 1) * dt)

    def clear_grid(self):
        self.grid_v_in = torch.zeros([self.N, n_grid, n_grid, dim], dtype=self.dtype, device=self.device)
        self.grid_m_in = torch.zeros([self.N, n_grid, n_grid], dtype=self.dtype, device=self.device)
        self.grid_v_out = torch.zeros([self.N, n_grid, n_grid, dim], dtype=self.dtype, device=self.device)

    def p2g(self, step):
        base = (self.x[step] * inv_dx - 0.5).floor().int()
        fx = self.x[step] * inv_dx - base.float()
        w = torch.cat([(0.5 * torch.pow(1.5 - fx, 2)).unsqueeze(2),
                       (0.75 - torch.pow(fx - 1, 2)).unsqueeze(2),
                       (0.5 * torch.pow(fx - 0.5, 2)).unsqueeze(2)], dim=2)

        new_F = (torch.eye(dim, dtype=self.dtype, device=self.device) + dt * self.C[step]) @ self.F[step]
        J = new_F.det()

        self.F[step + 1] = new_F
        r, s = utils.polar_decompose2d_ms(new_F)

        mass = 1
        cauchy = 2 * mu * (new_F - r) @ new_F.transpose(-1, -2)
        cauchy += torch.eye(2, dtype=self.dtype, device=self.device) \
                      .reshape((1, 1, 2, 2)).repeat(self.N, n_particles, 1, 1) * (la * J * (J - 1))[:, :, None, None]

        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * self.C[step]

        contr_idcs = torch.zeros([self.N, n_particles, 9, dim], dtype=torch.int64, device=self.device)
        v_in_contr = torch.zeros([self.N, n_particles, 9, dim], dtype=self.dtype, device=self.device)
        m_in_contr = torch.zeros([self.N, n_particles, 9], dtype=self.dtype, device=self.device)

        for i in range(3):
            for j in range(3):
                offset = torch.tensor([i, j], dtype=self.dtype, device=self.device)
                dpos = (offset - fx) * dx
                weight = w[:, :, i, 0] * w[:, :, j, 1]

                contr_idcs[:, :, i * 3 + j, :] = ((base + offset)[:, :, 0] * n_grid + (base + offset)[:, :, 1])[:, :,
                                                 None]
                v_in_contr[:, :, i * 3 + j] = weight[:, :, None] * (
                        mass * self.v[step] + (affine @ dpos.unsqueeze(-1)).squeeze(-1))
                m_in_contr[:, :, i * 3 + j] = weight * mass
                # self.grid_v_in[base[:, 0] + i, base[:, 1] + j] += weight[:, None] * (
                #         mass * self.v[step] + (affine @ dpos.unsqueeze(-1)).squeeze(-1))
                # self.grid_m_in[base[:, 0] + i, base[:, 1] + j] += weight * mass
        contr_idcs = contr_idcs.reshape(self.N, n_particles * 9, dim)
        v_in_contr = v_in_contr.reshape(self.N, n_particles * 9, dim)
        m_in_contr = m_in_contr.reshape(self.N, n_particles * 9)
        # Use scatter_add_ to add contributions to self.grid_v_in and self.grid_m_in
        self.grid_v_in.reshape(self.N, n_grid ** 2, -1).scatter_add_(dim=1, index=contr_idcs, src=v_in_contr). \
            reshape(self.N, n_grid, n_grid, 2)
        self.grid_m_in.reshape(self.N, n_grid ** 2).scatter_add_(dim=1, index=contr_idcs[:, :, 0],
                                                                 src=m_in_contr).reshape(self.N, n_grid, n_grid)

    def sdf(self, step, grid_pos):
        grid_xy = grid_pos * dx
        sphere_xy = self.sphere_x[:, step]
        return (torch.norm(grid_xy - sphere_xy[:, None, None, :], dim=-1) + 1e-8) - sphere_radius

    def normal(self, step, grid_pos):
        grid_xy = grid_pos * dx
        sphere_xy = self.sphere_x[:, step]
        return (grid_xy - sphere_xy[:, None, None, :]) / (
                torch.norm(grid_xy - sphere_xy[:, None, None, :], dim=-1)[:, :, :, None] + 1e-8)

    def collider_v(self, step, grid_pos, dt_):
        sphere_vel = self.sphere_v[:, step]

        return sphere_vel  # / dt_

    def collide(self, step, grid_pos, v_out, dt_):
        dist = self.sdf(step, grid_pos)
        influence = torch.min(torch.exp(-dist * softness), torch.tensor(1.0, dtype=self.dtype, device=self.device))

        mask = torch.logical_or(dist <= 0, influence > 0.1)

        D = self.normal(step, grid_pos)
        collider_v_at_grid = self.collider_v(step, grid_pos, dt_)

        input_v = v_out - collider_v_at_grid[:, None, None, :]
        normal_component = torch.einsum("bijk,bijk->bij", input_v, D)

        grid_v_t = input_v - (
                torch.min(normal_component, torch.tensor(0.0, dtype=self.dtype, device=self.device))[:, :, :,
                None] * D)
        grid_v_t_norm = torch.norm(grid_v_t, dim=-1) + 1e-8
        norm_mask = grid_v_t_norm > 1e-30

        grid_v_t_friction = grid_v_t / grid_v_t_norm[:, :, :, None] * torch.max(
            torch.tensor(0, dtype=self.dtype, device=self.device), grid_v_t_norm + normal_component * friction)[:, :, :,
                                                                      None]
        flag = (normal_component < 0).float()
        grid_v_t = (grid_v_t * (1.0 - norm_mask[:, :, :, None].float())) + (
                (grid_v_t_friction * flag[:, :, :, None] + grid_v_t * (1 - flag)[:, :, :, None]) * norm_mask[:, :, :,
                                                                                                   None].float())

        v_out_new = (collider_v_at_grid[:, None, None, :] + input_v * (1 - influence)[:, :, :, None] +
                     grid_v_t * influence[:, :, :, None]) * mask.float()[:, :, :, None] + \
                    v_out * (1.0 - mask.float())[:, :, :, None]
        return v_out_new

    def grid_op(self, step):
        # Create grid of indices.
        x_s = torch.arange(0, n_grid, dtype=self.dtype, device=self.device)
        y_s = torch.arange(0, n_grid, dtype=self.dtype, device=self.device)
        x, y = torch.meshgrid(x_s, y_s)
        grid_pos = torch.stack([x, y], dim=2).unsqueeze(0).repeat(self.N, 1, 1, 1)

        inv_m = 1 / (self.grid_m_in + eps)
        v_out = inv_m[:, :, :, None] * self.grid_v_in

        v_out_2 = self.collide(step, grid_pos, v_out, dt)

        zero_mask = torch.logical_and(grid_pos[:, :, :, 0] < bound, v_out_2[:, :, :, 0] < 0)
        zero_mask = torch.logical_or(zero_mask,
                                     torch.logical_and(grid_pos[:, :, :, 0] > n_grid - bound, v_out_2[:, :, :, 0] > 0))
        zero_mask = torch.logical_or(zero_mask,
                                     torch.logical_and(grid_pos[:, :, :, 1] > n_grid - bound, v_out_2[:, :, :, 1] > 0))
        v_out_3 = v_out_2 * (1.0 - zero_mask.float())[:, :, :, None]

        # Friction.
        friction_mask = torch.logical_and(grid_pos[:, :, :, 1] < bound, v_out_3[:, :, :, 1] < 0)
        normal = torch.tensor([0, 1], dtype=self.dtype, device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(
            0).repeat(self.N, n_grid, n_grid, 1)
        lin = torch.einsum("bijk,bijk->bij", v_out_3, normal)
        lin_mask = lin < 0

        vit = v_out_3 - lin[:, :, :, None] * normal
        lit = torch.norm(vit, dim=-1) + 1e-10

        lit_mask = lit + coeff * lin <= 0

        zero_lit_mask = torch.logical_and(friction_mask, torch.logical_and(lin_mask, lit_mask))
        v_out_4 = v_out_3 * (1.0 - zero_lit_mask.float())[:, :, :, None]

        non_zero_lit_mask = torch.logical_and(friction_mask,
                                              torch.logical_and(lin_mask, torch.logical_not(lit_mask)))
        v_out_5 = ((1 + coeff * lin / lit)[:, :, :, None] * vit) * non_zero_lit_mask.float()[:, :, :, None] + \
                  v_out_4 * (1.0 - non_zero_lit_mask.float())[:, :, :, None]

        self.grid_v_out = v_out_5

    def g2p(self, step):
        base = (self.x[step] * inv_dx - 0.5).floor().int()
        fx = self.x[step] * inv_dx - base.float()
        w = torch.cat([(0.5 * torch.pow(1.5 - fx, 2)).unsqueeze(2),
                       (0.75 - torch.pow(fx - 1, 2)).unsqueeze(2),
                       (0.5 * torch.pow(fx - 0.5, 2)).unsqueeze(2)], dim=2)

        new_v = torch.zeros([self.N, n_particles, 9, 2], dtype=self.dtype, device=self.device)
        new_C = torch.zeros([self.N, n_particles, 9, 2, 2], dtype=self.dtype, device=self.device)

        for i in range(3):
            for j in range(3):
                dpos = torch.tensor([i, j], dtype=self.dtype, device=self.device) - fx

                # Gather g_v.
                base_idx = ((base[:, :, 0] + i) * n_grid + (base[:, :, 1] + j)).unsqueeze(-1).repeat(1, 1, 2)
                g_v = self.grid_v_out.reshape(self.N, n_grid * n_grid, 2).gather(1, base_idx.long())
                # g_v = self.grid_v_out[:, base[:, :, 0] + i, base[:, :, 1] + j]

                weight = w[:, :, i, 0] * w[:, :, j, 1]

                new_v[:, :, 3 * i + j] = weight[:, :, None] * g_v
                new_C[:, :, 3 * i + j] = 4 * weight[:, :, None, None] * torch.einsum("kbi,kbj->kbij", g_v,
                                                                                     dpos) * inv_dx

        self.v[step + 1] = new_v.sum(dim=2)
        self.x[step + 1] = self.x[step] + dt * self.v[step + 1]
        self.C[step + 1] = new_C.sum(dim=2)

    def advance(self, step):
        self.clear_grid()
        self.p2g(step)
        self.grid_op(step)
        self.g2p(step)


def visualize(scene_: SceneBatchMultiShooting, out_dir: str = None, goal_x: torch.Tensor = None):
    if out_dir is not None:
        mmint_utils.make_dir(out_dir)
    plt.ion()
    fig, ax = plt.subplots()

    sphere_x = scene_.sphere_x.detach().cpu().numpy().reshape(-1, 2)

    for step in range(15, max_steps, 16):
        shot_idx = int(step / (max_steps / scene_.N))
        in_shot_idx = int(step % (max_steps / scene_.N))

        # for step in range(max_steps):
        ax.clear()
        plt.xlim(0.2, 0.8)
        plt.ylim(0.0, 0.6)
        circle = plt.Circle(
            (scene_.sphere_x.detach().cpu().numpy()[shot_idx, in_shot_idx][0],
             scene_.sphere_x.detach().cpu().numpy()[shot_idx, in_shot_idx][1]),
            sphere_radius, color='r')
        ax.add_patch(circle)
        ax.scatter(scene_.x[in_shot_idx].detach().cpu().numpy()[shot_idx, :, 0],
                   scene_.x[in_shot_idx].detach().cpu().numpy()[shot_idx, :, 1], c='b',
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


def loss(scene_: SceneBatchMultiShooting, goal_x_: torch.Tensor):
    loss_ = (scene_.x[-1] - goal_x_).norm(dim=-1).mean()
    return loss_


def get_init_from_scene_dict(scene_: SceneBatchMultiShooting, scene_dict_: dict):
    """
    Get initial conditions for multi shooting from an unrolling of a nominal
    trajectory on our scene.
    """
    x_all = scene_dict_["x"]
    v_all = scene_dict_["v"]
    C_all = scene_dict_["C"]
    F_all = scene_dict_["F"]
    sphere_x_all = scene_dict_["sphere_x"]
    sphere_v_all = scene_dict_["sphere_v"]

    device = scene_.device
    dtype = scene_.dtype

    N = scene_.N
    steps_per_shot = max_steps // N

    x_init = torch.zeros([N, n_particles, 2], dtype=dtype, device=device)
    v_init = torch.zeros([N, n_particles, 2], dtype=dtype, device=device)
    C_init = torch.zeros([N, n_particles, 2, 2], dtype=dtype, device=device)
    F_init = torch.zeros([N, n_particles, 2, 2], dtype=dtype, device=device)
    sphere_x_init = torch.zeros([N + 1, 2], dtype=dtype, device=device)

    for idx in range(N):
        x_init[idx] = x_all[idx * steps_per_shot]
        v_init[idx] = v_all[idx * steps_per_shot]
        C_init[idx] = C_all[idx * steps_per_shot]
        F_init[idx] = F_all[idx * steps_per_shot]

        sphere_x_init[idx] = sphere_x_all[idx * steps_per_shot]
    sphere_x_init[-1] = sphere_x_all[-1]

    return x_init, v_init, C_init, F_init, sphere_x_init


if __name__ == '__main__':
    # initialization
    scene = SceneBatchMultiShooting()

    # Load starting conditions for the sphere and deformable body.
    scene_dict = mmint_utils.load_gzip_pickle("scene.pkl.gzip")
    x_init_, v_init_, C_init_, F_init_, sphere_x_init_ = get_init_from_scene_dict(scene, scene_dict)

    scene.reset()
    scene.init_sphere_tensors(x_init_, v_init_, C_init_, F_init_, sphere_x_init_)

    scene_check = SceneBatch()
    sphere_end_pos = torch.tensor([0.5, 0.15], dtype=scene.dtype, device=scene.device)
    scene_check.reset()
    scene_check.init_sphere_tensors(sphere_end_pos)

    for s in trange(scene.steps_per_shot - 1):
        scene.advance(s)
        continue

    visualize(scene)
