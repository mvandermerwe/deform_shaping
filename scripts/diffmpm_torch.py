import argparse

import torch
from tqdm import trange

from deform_shaping import utils

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
max_steps = 2048
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


class Scene:

    def __init__(self):
        self.dtype = torch.float32
        self.device = torch.device("cpu")

        self.sphere_start_pos = torch.tensor([0.4, 0.2], dtype=self.dtype, device=self.device)
        self.sphere_end_pos = torch.tensor([0.6, 0.15], dtype=self.dtype, device=self.device)

        self.create_def_body()
        self.init_tensors()
        self.init_sphere_tensors()
        self.x[0] = self.x0

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
                    x_base + (i + 0.5) * real_dx,
                    y_base + (j + 0.5) * real_dy
                ])
        self.x0 = torch.tensor(x, dtype=self.dtype, device=self.device)
        n_particles = self.x0.shape[0]

    def init_tensors(self):
        self.x = torch.zeros([max_steps, n_particles, dim], dtype=self.dtype, device=self.device)
        self.v = torch.zeros([max_steps, n_particles, dim], dtype=self.dtype, device=self.device)
        self.C = torch.zeros([max_steps, n_particles, dim, dim], dtype=self.dtype, device=self.device)
        self.F = torch.zeros([max_steps, n_particles, dim, dim], dtype=self.dtype, device=self.device)
        self.sphere_x = torch.zeros([max_steps, dim], dtype=self.dtype, device=self.device)
        self.sphere_v = torch.zeros([max_steps, dim], dtype=self.dtype, device=self.device)

        self.grid_v_in = torch.zeros([n_grid, n_grid, dim], dtype=self.dtype, device=self.device)
        self.grid_m_in = torch.zeros([n_grid, n_grid], dtype=self.dtype, device=self.device)
        self.grid_v_out = torch.zeros([n_grid, n_grid, dim], dtype=self.dtype, device=self.device)

    def init_sphere_tensors(self):
        step_idx = torch.arange(max_steps, dtype=self.dtype, device=self.device)
        self.sphere_x[:, 0] = torch.lerp(self.sphere_start_pos[0], self.sphere_end_pos[0], step_idx / (max_steps - 1))
        self.sphere_x[:, 1] = torch.lerp(self.sphere_start_pos[1], self.sphere_end_pos[1], step_idx / (max_steps - 1))
        self.sphere_v[:-1, :] = (self.sphere_x[1:] - self.sphere_x[:-1]) / dt

    def clear_grid(self):
        self.grid_v_in.zero_()
        self.grid_m_in.zero_()

    def p2g(self, step):
        for p in range(n_particles):
            base = (self.x[step, p] * inv_dx - 0.5).floor().int()
            fx = self.x[step, p] * inv_dx - base.float()
            w = [0.5 * torch.pow(1.5 - fx, 2), 0.75 - torch.pow(fx - 1, 2), 0.5 * torch.pow(fx - 0.5, 2)]

            new_F = (torch.eye(dim, dtype=self.dtype, device=self.device) + dt * self.C[step, p]) @ self.F[step, p]
            J = new_F.det()

            self.F[step + 1, p] = new_F
            r, s = utils.polar_decompose(new_F)

            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.T + torch.diag(
                torch.tensor([2, la * (J - 1) * J], dtype=self.dtype, device=self.device))

            stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
            affine = stress + mass * self.C[step, p]

            for i in range(3):
                for j in range(3):
                    offset = torch.tensor([i, j], dtype=self.dtype, device=self.device)
                    dpos = (offset - fx) * dx
                    weight = w[i][0] * w[j][1]
                    self.grid_v_in[base[0] + i, base[1] + j] += weight * (mass * self.v[step, p] + affine @ dpos)
                    self.grid_m_in[base[0] + i, base[1] + j] += weight * mass

    def sdf(self, step, grid_pos):
        grid_xy = torch.tensor([grid_pos[0] * dx, grid_pos[1] * dx], dtype=self.dtype, device=self.device)
        sphere_xy = self.sphere_x[step]
        return torch.norm(grid_xy - sphere_xy) - sphere_radius

    def normal(self, step, grid_pos):
        grid_xy = torch.tensor([grid_pos[0] * dx, grid_pos[1] * dx], dtype=self.dtype, device=self.device)
        sphere_xy = self.sphere_x[step]
        return (grid_xy - sphere_xy) / torch.norm(grid_xy - sphere_xy)

    def collider_v(self, step, grid_pos, dt_):
        grid_xy = torch.tensor([grid_pos[0] * dx, grid_pos[1] * dx], dtype=self.dtype, device=self.device)
        sphere_xy = self.sphere_x[step]
        sphere_vel = self.sphere_v[step]

        return sphere_vel / dt_

    def collide(self, step, grid_pos, v_out, dt_):
        dist = self.sdf(step, grid_pos)
        influence = torch.min(torch.exp(-dist * softness), torch.tensor(1.0, dtype=self.dtype, device=self.device))

        if (softness > 0 and influence > 0.1) or dist <= 0:
            D = self.normal(step, grid_pos)
            collider_v_at_grid = self.collider_v(step, grid_pos, dt_)

            input_v = v_out - collider_v_at_grid
            normal_component = input_v.dot(D)

            grid_v_t = input_v - min(normal_component, 0) * D

            grid_v_t_norm = torch.norm(grid_v_t)
            grid_v_t_friction = grid_v_t / grid_v_t_norm * max(0, grid_v_t_norm + normal_component * friction)
            flag = (normal_component < 0 and torch.sqrt(grid_v_t.dot(grid_v_t)) > 1e-30).float()
            grid_v_t = grid_v_t_friction * flag + grid_v_t * (1 - flag)
            v_out = collider_v_at_grid + input_v * (1 - influence) + grid_v_t * influence

        return v_out

    def grid_op(self, step):
        for i in range(n_grid):
            for j in range(n_grid):
                inv_m = 1 / (self.grid_m_in[i, j] + eps)
                v_out = inv_m * self.grid_v_in[i, j]
                v_out[1] -= dt * gravity

                v_out = self.collide(step, torch.tensor([i, j], dtype=self.dtype, device=self.device), v_out, dt)

                if i < bound and v_out[0] < 0:
                    v_out[0] = 0
                    v_out[1] = 0
                if i > n_grid - bound and v_out[0] > 0:
                    v_out[0] = 0
                    v_out[1] = 0
                if j < bound and v_out[1] < 0:
                    v_out[0] = 0
                    v_out[1] = 0
                    normal = torch.tensor([0, 1], dtype=self.dtype, device=self.device)
                    lsq = (normal ** 2).sum()
                    if lsq > 0.5:
                        if coeff < 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:  # Friction.
                            lin = v_out.dot(normal)
                            if lin < 0:
                                vit = v_out - lin * normal
                                lit = vit.norm() + 1e-10
                                if lit + coeff * lin <= 0:
                                    v_out[0] = 0
                                    v_out[1] = 0
                                else:
                                    v_out = (1 + coeff * lin / lit) * vit
                if j > n_grid - bound and v_out[1] > 0:
                    v_out[0] = 0
                    v_out[1] = 0

                self.grid_v_out[i, j] = v_out

    def g2p(self, step):
        for p in range(n_particles):
            base = (self.x[step, p] * inv_dx - 0.5).floor().int()
            fx = self.x[step, p] * inv_dx - base.float()
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = torch.zeros(2, dtype=self.dtype, device=self.device)
            new_C = torch.zeros(2, 2, dtype=self.dtype, device=self.device)

            for i in range(3):
                for j in range(3):
                    dpos = torch.tensor([i, j], dtype=self.dtype, device=self.device) - fx
                    g_v = self.grid_v_out[base[0] + i, base[1] + j]
                    weight = w[i][0] * w[j][1]
                    new_v += weight * g_v
                    new_C += 4 * weight * torch.ger(g_v, dpos) * inv_dx  # TODO: Check.

            self.v[step + 1, p] = new_v
            self.x[step + 1, p] = self.x[step, p] + dt * self.v[step + 1, p]
            self.C[step + 1, p] = new_C

    def advance(self, step):
        self.clear_grid()
        self.p2g(step)
        self.grid_op(step)
        self.g2p(step)


if __name__ == '__main__':
    # initialization
    scene = Scene()

    for s in trange(max_steps):
        scene.advance(s)
