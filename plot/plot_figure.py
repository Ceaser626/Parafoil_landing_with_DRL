import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from plot.plot_utils import set_font_type_size, load_data, lims


medium_size = 14
bigger_size = 16
plt.rc('font', size=medium_size)  # controls default text sizes
plt.rc('axes', titlesize=bigger_size)  # font size of the axes title
plt.rc('axes', labelsize=bigger_size)  # font size of the x and y labels
plt.rc('xtick', labelsize=bigger_size)  # font size of the tick labels
plt.rc('ytick', labelsize=bigger_size)  # font size of the tick labels
plt.rc('legend', fontsize=medium_size)  # legend font size
plt.rc('font', family='Times New Roman')


def plot_training(name='default'):
    set_font_type_size()
    start, end = 0, 1000

    with open(f'path/to/output_dir/progress_{name}.csv', newline='') as csvfile:
        ppo_data = np.loadtxt(csvfile, delimiter=",", skiprows=1)
    iteration = ppo_data[start:end, 0]
    average = ppo_data[start:end, 3]
    std = ppo_data[start:end, 4]
    fig1, ax = plt.subplots()
    cm = plt.get_cmap('tab20')
    smoothing_window = 3
    # 计算std
    ret_average_smoothed = pd.Series(average).rolling(smoothing_window, min_periods=smoothing_window).mean()
    c = pd.concat([pd.Series(average[0:smoothing_window]), ret_average_smoothed[smoothing_window:]])
    ret_std_smoothed = pd.Series(std).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # 画图
    ax.plot(iteration, average, color=cm.colors[0], clip_on=False)
    ax.fill_between(iteration, average - std, average + std, color=cm.colors[1])
    # 设置坐标注释
    ax.set_xlabel('Iteration', labelpad=6)
    ax.set_ylabel('Return', labelpad=6)
    # 设置坐标轴
    ax.set_xlim(0, end)
    ax.set_ylim(-200, 100)
    # 保存
    fig1.tight_layout()
    plt.savefig(f'figure/Figure_1a.png')

    # read profile
    with open(f'path/to/output_dir/progress_{name}.csv', newline='') as csvfile:
        ppo_data = np.loadtxt(csvfile, delimiter=",", skiprows=1)
    iteration = ppo_data[start:end, 0]
    Explained_var = ppo_data[start:end, 16]
    Entropy = ppo_data[start:end, 17]
    fig2, ax1 = plt.subplots()
    cm = plt.get_cmap('tab10')
    # 画图
    ax1.plot(iteration, Explained_var, color=cm.colors[0], clip_on=False, label='Explained Var')
    # 设置坐标注释
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Explained Var', color=cm.colors[0])
    # axis-2
    ax2 = ax1.twinx()
    ax2.plot(iteration, Entropy, color=cm.colors[3], clip_on=False, label='Entropy')
    # 设置坐标注释
    ax2.set_ylabel('Entropy', labelpad=6, color=cm.colors[3])
    # 设置坐标轴
    ax1.set_xlim(0, end)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(-3, 4)
    ax1.tick_params(axis='y', colors=cm.colors[0])
    ax2.tick_params(axis='y', colors=cm.colors[3])
    # 保存
    fig1.tight_layout()
    plt.savefig(f'figure/Figure_1b.png')

    # read profile
    with open(f'path/to/output_dir/progress_{name}.csv', newline='') as csvfile:
        ppo_data = np.loadtxt(csvfile, delimiter=",", skiprows=1)
    iteration = ppo_data[start:end, 0]
    average = ppo_data[start:end, 6]
    std = ppo_data[start:end, 7]
    fig3, ax = plt.subplots()
    # 画图
    ax.plot(iteration, average, color=cm.colors[0], clip_on=False, label='Mean')
    ax.plot(iteration, std, color=cm.colors[3], clip_on=False, label='Std')
    # 设置坐标注释
    ax.set_xlabel('Iteration', labelpad=6)
    ax.set_ylabel('Terminal position error, m', labelpad=6)
    # 设置坐标轴
    ax.set_xlim(0, end)
    ax.set_ylim(0, 2500)
    plt.legend()
    # 保存
    fig3.tight_layout()
    plt.savefig(f'figure/Figure_1c.png')

    # read profile
    with open(f'path/to/output_dir/progress_{name}.csv', newline='') as csvfile:
        ppo_data = np.loadtxt(csvfile, delimiter=",", skiprows=1)
    iteration = ppo_data[start:end, 0]
    average = ppo_data[start:end, 8]
    std = ppo_data[start:end, 9]
    fig4, ax = plt.subplots()
    # 画图
    ax.plot(iteration, average, color=cm.colors[0], clip_on=False, label='Mean')
    ax.plot(iteration, std, color=cm.colors[3], clip_on=False, label='Std')
    # 设置坐标注释
    ax.set_xlabel('Iteration', labelpad=6)
    ax.set_ylabel('Terminal heading angle error, m', labelpad=6)
    # 设置坐标轴
    ax.set_xlim(0, end)
    ax.set_ylim(0, 180)
    plt.legend()
    # 保存
    fig4.tight_layout()
    plt.savefig(f'figure/Figure_1d.png')
    plt.show()


def plot_traj():
    cm = plt.get_cmap('tab10')
    set_font_type_size()
    traj = load_data(f'path/guidance_traj')

    # Trajectory
    # position profile
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(traj.t, traj.r_x, color=cm.colors[0], label=r'$r_x$', ls='-')
    ax1.plot(traj.t, traj.r_y, color=cm.colors[3], label=r'$r_y$', ls='-.')
    ax1.plot(traj.t, traj.r_z, color=cm.colors[2], label=r'$r_z$', ls='--')
    # 设置图注样式
    legend = ax1.legend(loc='best', fancybox=True, edgecolor='black', framealpha=1)
    # 设置坐标注释
    plt.xlabel('Time, s', labelpad=6)
    ax1.set_ylabel('Position, m', labelpad=6)
    # 设置坐标轴
    ax1.set_xlim(0, 110)
    ax1.set_ylim(-1000, 1500)
    # 保存
    plt.tight_layout()
    plt.savefig(f'figure/Figure_2a.png')

    # angle profile
    fig, ax2 = plt.subplots(1, 1)
    ax2.plot(traj.t, np.degrees(traj.phi.astype(float)), color=cm.colors[0], label=r'$\phi$', ls='-')
    ax2.plot(traj.t, np.degrees(traj.theta.astype(float)), color=cm.colors[3], label=r'$\theta$', ls='-.')
    ax2.plot(traj.t, np.degrees(traj.psi.astype(float)), color=cm.colors[2], label=r'$\psi$', ls='--')
    # 设置图注样式
    legend = ax2.legend(loc='best', fancybox=True, edgecolor='black', framealpha=1)
    # 设置坐标注释
    plt.xlabel('Time, s', labelpad=6)
    ax1.set_ylabel('Angle, deg', labelpad=6)
    # 设置坐标轴
    ax2.set_xlim(0, 110)
    ax2.set_ylim(-20, 200)
    # 保存
    plt.tight_layout()
    plt.savefig(f'figure/Figure_2b.tif.png')

    # print information
    print(f'Const wind: {traj.w_x[0]}, Flight time: {traj.t[-1]}, Terminal position: {traj.r_x[-1]}, {traj.r_y[-1]}')

    # velocity profile
    fig, ax3 = plt.subplots(1, 1)
    ax3.plot(traj.t, traj.v_x, color=cm.colors[0], label=r'$v_x$', ls='-')
    ax3.plot(traj.t, traj.v_y, color=cm.colors[3], label=r'$v_y$', ls='-.')
    ax3.plot(traj.t, traj.v_z, color=cm.colors[2], label=r'$v_z$', ls='--')
    # 设置图注样式
    legend = ax3.legend(loc='best', fancybox=True, edgecolor='black', framealpha=1)
    # 设置坐标注释
    plt.xlabel('Time, s', labelpad=6)
    ax3.set_ylabel('Velocity, m/s', labelpad=6)
    # 设置坐标轴
    ax3.set_xlim(0, 110)
    ax3.set_ylim(-10, 25)
    # 保存
    plt.tight_layout()
    plt.savefig(f'figure/Figure_2c.png')

    # angular velocity profile
    fig, ax4 = plt.subplots(1, 1)
    ax4.plot(traj.t, [np.rad2deg(i) for i in traj.o_x], label=r'$\omega _x$', color=cm.colors[0], ls='-')
    ax4.plot(traj.t, [np.rad2deg(i) for i in traj.o_y], label=r'$\omega _y$', color=cm.colors[3], ls='-.')
    ax4.plot(traj.t, [np.rad2deg(i) for i in traj.o_z], label=r'$\omega _z$', color=cm.colors[2], ls='--')
    # 设置图注样式
    legend = ax4.legend(loc='best', fancybox=True, edgecolor='black', framealpha=1)
    # 设置坐标注释
    plt.xlabel('Time, s', labelpad=6)
    ax4.set_ylabel('Angular velocity, deg/s', labelpad=6)
    # 设置坐标轴
    ax4.set_xlim(0, 110)
    ax4.set_ylim(-10, 10)
    ax4.set_yticks([-10, -5, 0, 5, 10])
    # 保存
    plt.tight_layout()
    plt.savefig(f'figure/Figure_2d.png')

    # control profile
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(traj.t, np.clip(traj.delta_l, 0, 1), color=cm.colors[0], ls='-', clip_on=False)
    ax2.plot(traj.t, np.clip(traj.delta_r, 0, 1), color=cm.colors[3], ls='-', clip_on=False)
    # 设置坐标注释
    plt.xlabel('Time, s', labelpad=6)
    ax1.set_ylabel('Left flap deflection', labelpad=6)
    ax2.set_ylabel('Right flap deflection', labelpad=6)
    # 设置坐标轴
    ax1.set_xlim(0, 110)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0, 110)
    ax2.set_ylim(0, 1)
    # 保存
    plt.tight_layout()
    plt.savefig(f'figure/Figure_3.png')
    plt.show()


def plot_mc():
    cm = plt.get_cmap('tab10')
    set_font_type_size()
    data = np.load(f'path/mc_traj.npz', allow_pickle=True)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    t = data['t']
    r_x = data['r_x']
    r_y = data['r_y']
    r_z = data['r_z']
    for i in range(100):
        r_x_slice = r_x[i]
        r_y_slice = r_y[i]
        r_z_slice = r_z[i]
        ax.plot3D(r_y_slice, r_x_slice, r_z_slice, linewidth=1)
    ax.grid(linestyle='--')
    # 设置坐标注释
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel('Crossrange, m', labelpad=6)
    ax.set_ylabel('Downrange, m', labelpad=6)
    ax.set_zlabel('Altitude, m', rotation=90, labelpad=6)
    # 设置坐标轴
    ax.set_xlim(-1000, 200)
    ax.set_ylim(-400, 800)
    ax.set_zlim(0, 1200)
    # 设置z轴位置
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)
    # 设置比例
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    # 设置边框
    xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
    i = np.array([xlims[0], ylims[0], zlims[0]])
    f = np.array([xlims[0], ylims[0], zlims[1]])
    p = art3d.Poly3DCollection(np.array([[i, f]]))
    p.set_color('black')
    ax.add_collection3d(p)
    ax.xaxis.pane.set_edgecolor('#000000')
    ax.yaxis.pane.set_edgecolor('#000000')
    ax.zaxis.pane.set_edgecolor('#000000')
    ax.xaxis.pane.set_alpha(1)
    ax.yaxis.pane.set_alpha(1)
    ax.zaxis.pane.set_alpha(1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # 设置尺寸
    fig.set_size_inches(6.4, 6)
    # 调整边距
    fig.set_tight_layout(True)
    # 保存
    fig.savefig(f'figure/Figure_4a.png')

    fig, ax = plt.subplots()
    t = data['t']
    r_x = data['r_x']
    r_y = data['r_y']
    psi = data['psi']
    r_x_f = []
    r_y_f = []
    psi_f = []
    d = []
    for i in range(100):
        r_x_slice = r_x[i]
        r_y_slice = r_y[i]
        psi_slice = np.degrees(psi[i]) - 180
        r_x_f.append(r_x_slice[-1])
        r_y_f.append(r_y_slice[-1])
        psi_f.append(psi_slice[-1])
        d.append(np.sqrt(r_x_slice[-1] ** 2 + r_y_slice[-1] ** 2))
    d.sort()
    cep_fifty = d[int(len(d) * 0.50)]
    cep_ninty = d[int(len(d) * 0.90)]
    angle = np.linspace(0, 2 * np.pi, 150)
    plt.scatter(r_y_f, r_x_f, color=cm.colors[3], alpha=0.5)
    plt.plot(cep_fifty * np.cos(angle), cep_fifty * np.sin(angle), label='50% CEP', color='k', ls='--')
    plt.plot(cep_ninty * np.cos(angle), cep_ninty * np.sin(angle), label='90% CEP', color='k', ls='-.')
    # 设置图注样式
    legend = ax.legend(loc='best', fancybox=True, edgecolor='black', framealpha=1)
    # 设置坐标边界
    plt.xlim([-300, 300])
    plt.ylim([-300, 300])
    # 设置坐标注释
    plt.xlabel('Crossrange, m', labelpad=6)
    plt.ylabel('Downrange, m', labelpad=6)
    # 设置比例
    ax.set_aspect('equal', adjustable='box')
    # 保存
    fig.tight_layout()
    plt.savefig(f'figure/Figure_4b.png')
    # print
    r_x_f = [abs(i) for i in r_x_f]
    r_y_f = [abs(i) for i in r_y_f]
    psi_f = [abs(i) for i in psi_f]
    print(f'50% CEP: {cep_fifty}; 90% CEP: {cep_ninty}')
    print(f'Downrange: mean-{np.mean(r_x_f):.2f}, std-{np.std(r_x_f):.2f}, min-{np.min(r_x_f):.2f}, max-{np.max(r_x_f):.2f}')
    print(f'Crossrange: mean-{np.mean(r_y_f):.2f}, std-{np.std(r_y_f):.2f}, min-{np.min(r_y_f):.2f}, max-{np.max(r_y_f):.2f}')
    print(f'Heading angle: mean-{np.mean(psi_f):.2f}, std-{np.std(psi_f):.2f}, min-{np.min(psi_f):.2f}, max-{np.max(psi_f):.2f}')
    plt.show()
