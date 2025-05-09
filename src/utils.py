import pinocchio as pin
import numpy as np


def figure_8(
    x_amplitude=0.4,
    z_amplitude=0.5,
    pos_offset=[0.0, 0.6, 0.5],
    timestep=0.01,
    period=10,
    num_periods=10,
    theta=0.0,
    axis="z",
):
    x = lambda t: pos_offset[0] + x_amplitude * np.sin(t)  # [-x_amplitude, x_amplitude]
    y = lambda t: pos_offset[1]
    z = (
        lambda t: pos_offset[2] + z_amplitude * np.sin(2 * t) / 2 + z_amplitude / 2
    )  # [-z_amplitude/2, z_amplitude/2]

    # rotation matrix about specified axis
    match axis:
        case "x":
            R = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(theta), -np.sin(theta)],
                    [0.0, np.sin(theta), np.cos(theta)],
                ]
            )
        case "y":
            R = np.array(
                [
                    [np.cos(theta), 0.0, np.sin(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(theta), 0.0, np.cos(theta)],
                ]
            )
        case "z":
            R = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

    def fig_8(t):
        unrot = np.array([x(t), y(t), z(t)])
        rot = R @ unrot
        return np.array([rot[0], rot[1], rot[2], 0.0, 0.0, 0.0])

    num_steps = int(period / timestep)
    traj = np.array([fig_8(t) for t in np.linspace(0, 2 * np.pi, num_steps)]).flatten()
    return np.tile(traj, num_periods)


# rk4 integrator
def rk4(model, data, q, v, u, dt):
    k1q = v
    k1v = pin.aba(model, data, q, v, u)
    q2 = pin.integrate(model, q, k1q * dt / 2)
    k2q = v + k1v * dt / 2
    k2v = pin.aba(model, data, q2, k2q, u)
    q3 = pin.integrate(model, q, k2q * dt / 2)
    k3q = v + k2v * dt / 2
    k3v = pin.aba(model, data, q3, k3q, u)
    q4 = pin.integrate(model, q, k3q * dt)
    k4q = v + k3v * dt
    k4v = pin.aba(model, data, q4, k4q, u)
    v_next = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    avg_v = (k1q + 2 * k2q + 2 * k3q + k4q) / 6
    q_next = pin.integrate(model, q, avg_v * dt)
    return q_next, v_next


# rk4 integrator with external wrench
def rk4(model, data, q, v, u, dt, f_ext=None):
    k1q = v
    k1v = pin.aba(model, data, q, v, u, f_ext)
    q2 = pin.integrate(model, q, k1q * dt / 2)
    k2q = v + k1v * dt / 2
    k2v = pin.aba(model, data, q2, k2q, u, f_ext)
    q3 = pin.integrate(model, q, k2q * dt / 2)
    k3q = v + k2v * dt / 2
    k3v = pin.aba(model, data, q3, k3q, u, f_ext)
    q4 = pin.integrate(model, q, k3q * dt)
    k4q = v + k3v * dt
    k4v = pin.aba(model, data, q4, k4q, u, f_ext)
    v_next = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    avg_v = (k1q + 2 * k2q + 2 * k3q + k4q) / 6
    q_next = pin.integrate(model, q, avg_v * dt)
    return q_next, v_next
