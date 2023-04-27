import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)
# ti.init(default_fp=real, arch=ti.cpu, flatten_if=True, cpu_max_num_threads=1)

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
max_steps = 1024
steps = 1024
gravity = 0.0
target = [0.8, 0.2]

sphere_start_pos = [0.5, 0.17]
sphere_end_pos = [0.5, 0.15]
# sphere_start_pos = [0.4, 0.2]
# sphere_end_pos = [0.6, 0.15]
sphere_radius = 0.05

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()
sphere_pos = vec()
sphere_vel = vec()

loss = scalar()

n_sin_waves = 4
# weights = scalar()
# bias = scalar()
x_avg = vec()
#
# actuation = scalar()
actuation_omega = 20
act_strength = 4


def sphere_position(s, total_steps):
    sphere_x = sphere_start_pos[0] * (1 - (s / total_steps)) + sphere_end_pos[0] * (s / total_steps)
    sphere_y = sphere_start_pos[1] * (1 - (s / total_steps)) + sphere_end_pos[1] * (s / total_steps)
    return [sphere_x, sphere_y]


def allocate_fields():
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.dense(ti.i, max_steps).place(sphere_pos)
    ti.root.dense(ti.i, max_steps).place(sphere_vel)
    ti.root.place(loss, x_avg)

    ti.root.lazy_grad()


@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        # print("base:", base)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        # print("fx", fx)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # print("w", w)
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        # print("new_F", new_F)
        J = (new_F).determinant()
        # print("J", J)

        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)
        # print("r", r)
        # print("s", s)

        act_id = actuator_id[p]

        # act = actuation[f, ti.max(0, act_id)] * act_strength
        act = 0.0
        # ti.#print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy_add = ti.Matrix.diag(2, la * (J - 1) * J)
            # print("cauchy_add", cauchy_add)
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + cauchy_add
            # print("cauchy_in", cauchy)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]

        # print("cauchy", cauchy)
        # print("stress", stress)
        # print("affine", affine)

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5
softness = 666.0  # From PlasticineLab
friction = 2.0  # ?


@ti.func
def sdf(f, grid_pos):
    grid_xy = ti.Vector([grid_pos[0] * dx, grid_pos[1] * dx])
    sphere_xy = sphere_pos[f]
    return length(grid_xy - sphere_xy) - sphere_radius


@ti.func
def normal(f, grid_pos):
    grid_xy = ti.Vector([grid_pos[0] * dx, grid_pos[1] * dx])
    sphere_xy = sphere_pos[f]
    return (grid_xy - sphere_xy) / length(grid_xy - sphere_xy)


@ti.func
def collider_v(f, grid_pos, dt_):
    grid_xy = ti.Vector([grid_pos[0] * dx, grid_pos[1] * dx])
    sphere_xy = sphere_pos[f]
    sphere_vel_ = sphere_vel[f]

    rel_pos = grid_xy - sphere_xy

    # next_sphere_xy = sphere_xy + (sphere_vel_ * dt_)
    # next_rel_pos = next_sphere_xy + rel_pos
    # return (next_rel_pos - grid_xy) / dt_

    return sphere_vel_ / dt_


@ti.func
def length(x_):
    return ti.sqrt(x_.dot(x_) + 1e-8)


@ti.func
def collide(f, grid_pos, v_out, dt):
    dist = sdf(f, grid_pos)
    # print("dist", dist)
    influence = ti.min(ti.exp(-dist * softness), 1)
    # print("influence", influence)
    if (softness > 0 and influence > 0.1) or dist <= 0:
        D = normal(f, grid_pos)
        # print("D", D)
        collider_v_at_grid = collider_v(f, grid_pos, dt)
        # print("collider_v_at_grid", collider_v_at_grid)

        input_v = v_out - collider_v_at_grid
        normal_component = input_v.dot(D)

        grid_v_t = input_v - ti.min(normal_component, 0) * D
        # print("grid_v_t", grid_v_t)

        grid_v_t_norm = length(grid_v_t)
        # print("grid_v_t_norm", grid_v_t_norm)
        grid_v_t_friction = grid_v_t / grid_v_t_norm * ti.max(0, grid_v_t_norm + normal_component * friction)
        flag = ti.cast(normal_component < 0 and ti.sqrt(grid_v_t.dot(grid_v_t)) > 1e-30, ti.f32)
        grid_v_t = grid_v_t_friction * flag + grid_v_t * (1 - flag)
        v_out = collider_v_at_grid + input_v * (1 - influence) + grid_v_t * influence

        # print(self.position[f], f)
        # print(grid_pos, collider_v, v_out, dist, self.friction, D)
        # if v_out[1] > 1000:
        # print(input_v, collider_v_at_grid, normal_component, D)

    return v_out


@ti.kernel
def grid_op(s: ti.i32):
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity

        # print("v_out (pre-collide)", v_out)

        v_out = collide(s, ti.Vector([i, j]), v_out, dt)
        # print("v_out (post-collide)", v_out)

        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal ** 2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:  # Friction
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

        # print("v_out (final)", v_out)

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        # print("base", base)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        # print("fx", fx)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        # print("w", w)
        new_v = ti.Vector([0.0, 0.0])
        # print("new_v", new_v)
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        # print("new_C", new_C)

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        # print("new_v", new_v)
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        # print("x", x[f + 1, p])
        C[f + 1, p] = new_C
        # print("new_C", new_C)


@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])


@ti.kernel
def compute_loss():
    dist = x_avg[None][0]
    loss[None] = -dist


@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    p2g(s)
    grid_op(s)
    g2p(s)
    pass


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op(s)

    g2p.grad(s)
    grid_op.grad(s)
    p2g.grad(s)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act


def robot(scene):
    scene.set_offset(0.5 - 0.15, 0.0)
    scene.add_rect(0.0, 0.019, 0.3, 0.1, -1, ptype=1)
    scene.set_n_actuators(0)


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder):
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    sphere_pos_ = sphere_pos.to_numpy()[s]
    radius_in_pixels = sphere_radius * 640.0
    gui.circle(sphere_pos_, color=0xFF0000, radius=radius_in_pixels)
    for i in range(n_particles):
        color = 0x111111
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    # gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    options = parser.parse_args()

    # initialization
    scene = Scene()
    robot(scene)
    scene.finalize()
    allocate_fields()

    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

    for i in range(max_steps):
        sphere_pos[i] = sphere_position(i, max_steps)
        sphere_vel[i] = (np.array(sphere_end_pos) - np.array(sphere_start_pos)) / (max_steps - 1)

    # visualize
    forward(max_steps)
    for s in range(15, max_steps, 16):
        visualize(s, 'diffmpm/iter{:03d}/'.format(0))

    # losses = []
    # for iter in range(options.iters):
    #     with ti.ad.Tape(loss):
    #         forward()
    #     l = loss[None]
    #     losses.append(l)
    #     print('i=', iter, 'loss=', l)
    #     learning_rate = 0.1
    #
    #     if iter % 10 == 0:
    #         # visualize
    #         forward(1500)
    #         for s in range(15, 1500, 16):
    #             visualize(s, 'diffmpm/iter{:03d}/'.format(iter))
    #
    # # ti.profiler_print()
    # plt.title("Optimization of Initial Velocity")
    # plt.ylabel("Loss")
    # plt.xlabel("Gradient Descent Iterations")
    # plt.plot(losses)
    # plt.show()


if __name__ == '__main__':
    main()
