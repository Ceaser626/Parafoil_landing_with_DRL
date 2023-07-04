import numpy as np
from scipy import integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.tools import SaveStatus


def set_font_type_size():
    small_size = 12
    medium_size = 14
    bigger_size = 16
    plt.rc('font', size=medium_size)  # controls default text sizes
    plt.rc('axes', titlesize=bigger_size)  # font size of the axes title
    plt.rc('axes', labelsize=bigger_size)  # font size of the x and y labels
    plt.rc('xtick', labelsize=bigger_size)  # font size of the tick labels
    plt.rc('ytick', labelsize=bigger_size)  # font size of the tick labels
    plt.rc('legend', fontsize=small_size)  # legend font size
    plt.rc('font', family='Times New Roman')


def load_data(name, mode='6-DOF'):
    data = np.load(f'{name}.npz', allow_pickle=True)

    if mode == '6-DOF':
        traj = SaveStatus()
        traj.t = data['t']
        traj.r_x = data['r_x']
        traj.r_y = data['r_y']
        traj.r_z = data['r_z']
        traj.phi = data['phi']
        traj.theta = data['theta']
        traj.psi = data['psi']
        traj.v_x = data['v_x']
        traj.v_y = data['v_y']
        traj.v_z = data['v_z']
        traj.o_x = data['o_x']
        traj.o_y = data['o_y']
        traj.o_z = data['o_z']
        traj.w_x = data['w_x']
        traj.delta_l = data['delta_l']
        traj.delta_r = data['delta_r']
        traj.delta_a = data['delta_a']
        traj.delta_s = data['delta_s']

    return traj


def lims(mplotlims):
    scale = 1.021
    offset = (mplotlims[1] - mplotlims[0]) * scale
    return mplotlims[1] - offset, mplotlims[0] + offset
