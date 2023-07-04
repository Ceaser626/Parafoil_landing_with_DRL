import numpy as np
from pyomo.environ import *
from scipy.interpolate import interp1d


def np_norm(v):
    output = np.sqrt(v[0, 0] ** 2 + v[1, 0] ** 2 + v[2, 0] ** 2)
    return output


def interpolate(t_list, value_list):
    f_value = interp1d(t_list, value_list, fill_value="extrapolate")
    return f_value


def lgr_points(num):
    """
    Parameters
    ----------
    num: number of Flipped shifted-Radau points

    Returns
    -------
    tau_list: LGR points (0, 1]
    """
    if num == 2:
        tau_list = [0.333333333333333, 1]
    elif num == 3:
        tau_list = [0.155051025721682, 0.644948974278318, 1]
    elif num == 4:
        tau_list = [0.088587959512704, 0.409466864440735, 0.787659461760847, 1]
    elif num == 5:
        tau_list = [0.057104196114518, 0.276843013638124, 0.583590432368917, 0.860240135656220, 1]
    elif num == 6:
        tau_list = [0.039809857051469, 0.198013417873608, 0.437974810247386, 0.695464273353636, 0.901464914201174, 1]
    else:
        tau_list = []

    return tau_list


class SaveStatus:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t = []
        self.r_x = []
        self.r_y = []
        self.r_z = []
        self.phi = []
        self.theta = []
        self.psi = []
        self.v_x = []
        self.v_y = []
        self.v_z = []
        self.o_x = []
        self.o_y = []
        self.o_z = []
        self.w_x = []
        self.delta_l = []
        self.delta_r = []
        self.delta_a = []
        self.delta_s = []

    def store_env(self, t, obs, act):
        r_x, r_y, r_z, phi, theta, psi, v_x, v_y, v_z, o_x, o_y, o_z, w_x = obs
        delta_l, delta_r = act
        delta_a = delta_r - delta_l
        delta_s = min(delta_l, delta_r)

        self.t.append(t)
        self.r_x.append(r_x)
        self.r_y.append(r_y)
        self.r_z.append(r_z)
        self.phi.append(phi)
        self.theta.append(theta)
        self.psi.append(psi)
        self.v_x.append(v_x)
        self.v_y.append(v_y)
        self.v_z.append(v_z)
        self.o_x.append(o_x)
        self.o_y.append(o_y)
        self.o_z.append(o_z)
        self.w_x.append(w_x)
        self.delta_l.append(delta_l)
        self.delta_r.append(delta_r)
        self.delta_a.append(delta_a)
        self.delta_s.append(delta_s)

    def store_nlp(self, m):
        for i in m.tau:
            self.t.append(value(m.time[i]))
            self.r_x.append(value(m.r_x[i]))
            self.r_y.append(value(m.r_y[i]))
            self.r_z.append(value(m.r_z[i]))
            self.phi.append(value(m.phi[i]))
            self.theta.append(value(m.theta[i]))
            self.psi.append(value(m.psi[i]))
            self.v_x.append(value(m.v_x[i]))
            self.v_y.append(value(m.v_y[i]))
            self.v_z.append(value(m.v_z[i]))
            self.o_x.append(value(m.o_x[i]))
            self.o_y.append(value(m.o_y[i]))
            self.o_z.append(value(m.o_z[i]))
            self.delta_a.append(value(m.delta_a[i]))
            self.delta_s.append(value(m.delta_s[i]))
            if value(m.delta_a[i]) > 0:
                delta_l = value(m.delta_s[i])
                delta_r = value(m.delta_s[i]) + value(m.delta_a[i])
            else:
                delta_l = value(m.delta_s[i]) - value(m.delta_a[i])
                delta_r = value(m.delta_s[i])
            self.delta_l.append(delta_l)
            self.delta_r.append(delta_r)

    def store_mc(self, scene):  # scene is a complete trajectory
        self.t.append(scene.t)
        self.r_x.append(scene.r_x)
        self.r_y.append(scene.r_y)
        self.r_z.append(scene.r_z)
        self.phi.append(scene.phi)
        self.theta.append(scene.theta)
        self.psi.append(scene.psi)
        self.v_x.append(scene.v_x)
        self.v_y.append(scene.v_y)
        self.v_z.append(scene.v_z)
        self.o_x.append(scene.o_x)
        self.o_y.append(scene.o_y)
        self.o_z.append(scene.o_z)
        self.w_x.append(scene.w_x)
        self.delta_l.append(scene.delta_l)
        self.delta_r.append(scene.delta_r)
        self.delta_a.append(scene.delta_a)
        self.delta_s.append(scene.delta_s)

    def save(self, save_name):
        t = np.array(self.t, dtype=object)
        r_x = np.array(self.r_x, dtype=object)
        r_y = np.array(self.r_y, dtype=object)
        r_z = np.array(self.r_z, dtype=object)
        phi = np.array(self.phi, dtype=object)
        theta = np.array(self.theta, dtype=object)
        psi = np.array(self.psi, dtype=object)
        v_x = np.array(self.v_x, dtype=object)
        v_y = np.array(self.v_y, dtype=object)
        v_z = np.array(self.v_z, dtype=object)
        o_x = np.array(self.o_x, dtype=object)
        o_y = np.array(self.o_y, dtype=object)
        o_z = np.array(self.o_z, dtype=object)
        w_x = np.array(self.w_x, dtype=object)
        delta_l = np.array(self.delta_l, dtype=object)
        delta_r = np.array(self.delta_r, dtype=object)
        delta_a = np.array(self.delta_a, dtype=object)
        delta_s = np.array(self.delta_s, dtype=object)
        np.savez(f'path/{save_name}', t=t, r_x=r_x, r_y=r_y, r_z=r_z,
                 phi=phi, theta=theta, psi=psi, v_x=v_x, v_y=v_y, v_z=v_z, o_x=o_x, o_y=o_y, o_z=o_z,
                 w_x=w_x,
                 delta_l=delta_l, delta_r=delta_r, delta_a=delta_a, delta_s=delta_s)
