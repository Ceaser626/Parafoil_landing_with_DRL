import time
import random
import matplotlib.pyplot as plt
from env.parafoil_partial import ParafoilEnv
from RL.net import *
from utils.tools import SaveStatus, interpolate
from plot.plot_figure import plot_training, plot_traj, plot_mc


# 1. test env 2. test trained agent
def test_agent(seed=1, name='default'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    model = torch.load(f'path/to/output_dir/model_{name}.pth')
    env = ParafoilEnv()

    ep_ret, ep_len = 0, 0
    t, o = 0, env.reset(save_dryden=True)
    t_run = []
    env.dt = 2

    # plot figure
    ax1 = plt.axes(projection='3d')
    ax1.set_xlim(-1500, 1500)
    ax1.set_ylim(-1500, 1500)
    ax1.set_zlim(0, 1200)
    text = ax1.text(x=0.0, y=1.02, z=0, s='', fontsize=8)
    ax_list = [ax1, text]

    while True:
        env.frame(ax_list)
        plt.pause(1e-1)

        tic = time.perf_counter()
        action, _, _, _ = model.get_action_and_value(torch.Tensor(o), deterministic=True)
        toc = time.perf_counter()
        t_run.append(toc - tic)
        next_t = t + env.dt
        next_o, r, d, saved_status = env.step(action.numpy().squeeze())

        t = next_t
        o = next_o
        ep_ret += r
        print(r)
        ep_len += 1

        if d:
            unscaled_o = env.unscale(o)
            r_x, r_y, psi = unscaled_o[0], unscaled_o[1], unscaled_o[3]
            r_err = np.sqrt(r_x ** 2 + r_y ** 2)
            psi_err = np.rad2deg(abs(psi - np.pi))

            print(f'Position error: {r_err}, Psi error: {psi_err}')
            print(f'EpRet {ep_ret} EpLen {ep_len}')
            plt.show()

            saved_status.save(f'guidance_traj')
            t_run = np.array(t_run, dtype=object)
            np.savez(f'path/t_run', t_run=t_run)

            break


def monte_carlo(seed=1, num_mc=500, name='default'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    model = torch.load(f'path/to/output_dir/model_{name}.pth')
    env = ParafoilEnv()
    mc_status = SaveStatus()

    n, ep_ret, ep_len = 0, 0, 0
    t, o = 0, env.reset()

    while n < num_mc:
        action, _, _, _ = model.get_action_and_value(torch.Tensor(o), deterministic=True)
        next_t = t + env.dt
        next_o, r, d, saved_status = env.step(action.numpy().squeeze())

        t = next_t
        o = next_o
        ep_ret += r
        ep_len += 1

        if d:
            mc_status.store_mc(saved_status)
            n, ep_ret, ep_len = n + 1, 0, 0
            t, o, r, d = 0, env.reset(), 0, False

    mc_status.save(f'mc_traj')

    return mc_status


if __name__ == '__main__':
    mod = 0
    file_name = 'part_obs_u_dof_2_1000_iter'

    if mod == 0:
        plot_training(name=file_name)
        # test_agent(6, name=file_name)
        plot_traj()
    elif mod == 1:
        # monte_carlo(1, 100, name=file_name)
        plot_mc()
