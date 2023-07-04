from RL.ppo import ppo
from plot.plot_figure import plot_training


if __name__ == '__main__':
    file_name = 'part_obs_u_dof_2_1000_iter'
    ppo(name=file_name)
    plot_training(name=file_name)
