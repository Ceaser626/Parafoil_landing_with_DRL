import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def dryden_wind(t_len, dt, save=False, plot=False):
    """
    Parameters
    ----------
    t_len: time interval
    dt: integration interval
    plot: (flag)
    save: (flag)

    Returns
    -------
    u_disturb: (size = t_len / dt)
    v_disturb
    w_disturb
    """

    # parameter
    w_20 = 45 * 0.514444  # convert speed from 30 knots to meter per second
    altitude = 200
    air_speed = 18

    # coefficient
    L_u = altitude / ((0.177 + 0.000823 * altitude) ** 1.2)
    L_v = L_u
    L_w = altitude
    sigma_w = 0.1 * w_20
    sigma_u = sigma_w / ((0.177 + 0.000823 * altitude) ** 0.4)
    sigma_v = sigma_u

    # transfer function
    num_u = [sigma_u * np.sqrt(2 * L_u / np.pi / air_speed) * air_speed]
    den_u = [L_u, air_speed]
    H_u = signal.TransferFunction(num_u, den_u)

    b = sigma_v * np.sqrt(L_v / np.pi / air_speed)
    num_v = [np.sqrt(3) * L_v / air_speed * b, b]
    den_v = [(L_v / air_speed) ** 2, 2 * L_v / air_speed, 1]
    H_v = signal.TransferFunction(num_v, den_v)

    c = sigma_w * np.sqrt(L_w / np.pi / air_speed)
    num_w = [np.sqrt(3) * L_w / air_speed * c, c]
    den_w = [(L_w / air_speed) ** 2, 2 * L_w / air_speed, 1]
    H_w = signal.TransferFunction(num_w, den_w)

    # white gaussian noise
    num_samples = int(t_len / dt)
    t_p = np.linspace(0, t_len, num_samples)  # from 0-160s, interval 0.25s
    mean = 0
    std = 1
    wgn_input_u = np.random.normal(mean, std, size=num_samples)
    wgn_input_v = np.random.normal(mean, std, size=num_samples)
    wgn_input_w = np.random.normal(mean, std, size=num_samples)

    # dryden wind
    tout1, u_disturb, x1 = signal.lsim(H_u, wgn_input_u, t_p)
    tout2, v_disturb, x2 = signal.lsim(H_v, wgn_input_v, t_p)
    tout3, w_disturb, x3 = signal.lsim(H_w, wgn_input_w, t_p)

    # plot
    if plot:
        plt.figure(1)
        plt.plot(t_p, u_disturb, 'b')
        plt.ylabel('along-wind in m/s')
        plt.xlabel('time in seconds')
        plt.grid(True)

        plt.figure(2)
        plt.plot(t_p, v_disturb, 'r')
        plt.ylabel('cross-wind in m/s')
        plt.xlabel('time in seconds')
        plt.grid(True)

        plt.figure(3)
        plt.plot(t_p, w_disturb, 'g')
        plt.ylabel('vertical-wind in m/s')
        plt.xlabel('time in seconds')
        plt.grid(True)
        plt.show()

    # save
    if save:
        t_save = np.array(t_p, dtype=object)
        u_save = np.array(u_disturb, dtype=object)
        v_save = np.array(v_disturb, dtype=object)
        w_save = np.array(w_disturb, dtype=object)
        np.savez(f'path/dryden_wind', t_save=t_save, u_save=u_save, v_save=v_save, w_save=w_save)

    return t_p, u_disturb, v_disturb, w_disturb


if __name__ == '__main__':
    # test dryden wind
    u_disturb, v_disturb, w_disturb = dryden_wind(160, 0.25)
