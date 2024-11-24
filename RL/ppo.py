import argparse
import os
import random
import time, datetime
from distutils.util import strtobool

import csv

import numpy as np
import scipy.signal
from itertools import repeat
import copy
import torch.optim as optim
from multiprocessing import Pool, cpu_count
from env.parafoil_partial import ParafoilEnv
from RL.net import *


def parse_args(save=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--num-updates", type=int, default=1000,
                        help="the number of program iteration")
    parser.add_argument("--update-epochs", type=int, default=3,
                        help="the number of epoch to update the policy")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="the number of parallel worker")
    parser.add_argument("--num-minibatches", type=int, default=192,
                        help="the number of mini-batches")
    parser.add_argument("--minibatch-size", type=int, default=64,
                        help="the size of mini-batches")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.985,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.98,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.001,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.03,
                        help="the target KL divergence threshold")

    args = parser.parse_args()

    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_steps = int(args.batch_size // args.num_envs)

    if save:
        with open('path/to/output_dir/config.txt', 'w') as fw:
            print(f'num_updates:        {args.num_updates} \n', file=fw)
            print(f'update_epochs:      {args.update_epochs} \n', file=fw)
            print(f'batch_size:         {args.batch_size} \n', file=fw)
            print(f'num_steps:          {args.num_steps} \n', file=fw)
            print(f'num_minibatches:    {args.num_minibatches} \n', file=fw)
            print(f'minibatch_size:     {args.minibatch_size} \n', file=fw)
            print(f'learning_rate:      {args.learning_rate} \n', file=fw)
            print(f'gamma:              {args.gamma} \n', file=fw)
            print(f'gae_lambda:         {args.gae_lambda} \n', file=fw)
            print(f'clip_coef:          {args.clip_coef} \n', file=fw)
            print(f'ent_coef:           {args.ent_coef} \n', file=fw)
            print(f'vf_coef:            {args.vf_coef} \n', file=fw)
            print(f'max_grad_norm:      {args.max_grad_norm} \n', file=fw)
            print(f'target_kl:          {args.target_kl} \n', file=fw)

    return args


class PPOBuffer:

    def __init__(self, obs_dim, act_dim, size, args):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = args.gamma, args.gae_lambda
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.logp_buf[self.ptr] = logp
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cum_sum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cum_sum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.obs_buf, act=self.act_buf, logp=self.logp_buf,
                    val=self.val_buf, adv=self.adv_buf, ret=self.ret_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    @staticmethod
    def discount_cum_sum(x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def parallel_env(n, iter_num, name):

    args = parse_args()
    env = ParafoilEnv()
    num_steps = args.num_steps
    if iter_num == 1:
        agent = Agent(env)
    else:
        agent = torch.load(f'path/to/output_dir/model_{name}.pth')
    buf = PPOBuffer(env.obs_dimension, env.act_dimension, num_steps, args)

    obs = torch.Tensor(env.reset())
    ep_ret_list, ep_len_list, ep_err_list, ep_psi_list = [], [], [], []
    ep_ret, ep_len, success_num, sum_num = 0, 0, 0, 0

    for step in range(0, num_steps):

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs)

        next_o, reward, d, _ = env.step(action.numpy().squeeze())
        ep_ret = ep_ret + reward
        ep_len = ep_len + 1

        # save
        buf.store(obs.numpy().squeeze(), action.numpy().squeeze(), reward, value, logprob.numpy().squeeze())

        obs = torch.Tensor(next_o)

        trajectory_end = d
        steps_end = (step == num_steps-1)
        if trajectory_end or steps_end:
            if steps_end:
                with torch.no_grad():
                    _, _, _, value = agent.get_action_and_value(obs)
            else:
                value = 0
            buf.finish_path(value)

            if trajectory_end:
                ep_ret_list.append(ep_ret)
                ep_len_list.append(ep_len)
                sum_num = sum_num + 1
                terminal_o = env.unscale(obs.numpy().squeeze())
                x, y, psi = terminal_o[0], terminal_o[1], terminal_o[3]
                r_err = np.sqrt(x ** 2 + y ** 2)
                psi_err = abs(psi - np.pi)
                if (r_err <= env.D_rf_max) and (psi_err <= env.D_psif_max):
                    success_num = success_num + 1
                ep_err_list.append(r_err)
                ep_psi_list.append(np.degrees(psi_err))

                obs = torch.Tensor(env.reset())
                ep_ret, ep_len = 0, 0

    info = [ep_ret_list, ep_len_list, ep_err_list, ep_psi_list, success_num, sum_num]
    return buf, info


def ppo(name='default'):
    args = parse_args(save=True)

    # save to csv
    file = open(f'path/to/output_dir/progress_{name}.csv', 'w', newline='')
    header = ['Iteration', 'Step', 'Ep_num', 'EpRet_mean', 'EpRet_std', 'EpLen_mean',
              'EpErr_mean', 'EpErr_std', 'EpPsi_mean', 'EpPsi_std', 'EpSuccess_num', 'Value_loss', 'Policy_loss',
              'Old_approx_kl', 'Approx_kl', 'Clipfrac', 'Explained_variance', 'Entropy', 'Time']
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # RL update on gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # setup
    env = ParafoilEnv()
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, env.obs_dimension)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, env.act_dimension)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    advantages = torch.zeros((args.num_steps, args.num_envs)).to(device)
    returns = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # start the game
    global_step = 0
    iter_count = range(args.num_envs)

    for update in range(1, args.num_updates + 1):
        start_time = time.time()
        ep_ret_list, ep_len_list, ep_err_list, ep_psi_list = [], [], [], []
        success_num, sum_num = 0, 0
        # update_list = [update] * args.num_envs

        # annealing the rate
        frac = 1.0 - (update - 1.0) / args.num_updates
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

        # interact with env using parallel computing
        with Pool(processes=cpu_count()) as pool:
            data_env = pool.starmap(parallel_env, zip(iter_count, repeat(update), repeat(name)))
        global_step += args.num_envs * args.num_steps

        for i in range(args.num_envs):
            buf, info = data_env[i]
            buf_data = buf.get()
            obs[:, i, :] = buf_data['obs'].to(device)
            actions[:, i, :] = buf_data['act'].to(device)
            logprobs[:, i] = buf_data['logp'].to(device)
            values[:, i] = buf_data['val'].to(device)
            advantages[:, i] = buf_data['adv'].to(device)
            returns[:, i] = buf_data['ret'].to(device)
            ep_ret_list = ep_ret_list + info[0]
            ep_len_list = ep_len_list + info[1]
            ep_err_list = ep_err_list + info[2]
            ep_psi_list = ep_psi_list + info[3]
            success_num = success_num + info[4]
            sum_num = sum_num + info[5]

        # flatten the batch
        b_obs = obs.reshape((-1, env.obs_dimension))
        b_actions = actions.reshape((-1, env.act_dimension))
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()  # probability ratio

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # 归一化advantage
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-10)

                # Policy loss
                pg_loss1 = mb_advantages * ratio
                pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.min(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.reshape(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )  # clip value差值范围
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = - (pg_loss + args.ent_coef * entropy_loss - args.vf_coef * v_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    print('max kl')
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # save agent (with safeguard)
        if (update % 1) == 0:
            save_agent = copy.deepcopy(agent)
            torch.save(save_agent.to(torch.device("cpu")), f'path/to/output_dir/model_{name}.pth')

        # record info
        writer.writerow({'Iteration': update,
                         'Step': global_step,
                         'Ep_num': sum_num,
                         'EpRet_mean': np.mean(ep_ret_list),
                         'EpRet_std': np.std(ep_ret_list),
                         'EpLen_mean': np.mean(ep_len_list),
                         'EpErr_mean': np.mean(ep_err_list),
                         'EpErr_std': np.std(ep_err_list),
                         'EpPsi_mean': np.mean(ep_psi_list),
                         'EpPsi_std': np.std(ep_psi_list),
                         'EpSuccess_num': success_num,
                         'Value_loss': v_loss.item(),
                         'Policy_loss': pg_loss.item(),
                         'Old_approx_kl': old_approx_kl.item(),
                         'Approx_kl': approx_kl.item(),
                         'Clipfrac': np.mean(clipfracs),
                         'Explained_variance': explained_var,
                         'Entropy': entropy_loss.item(),
                         'Time': time.time() - start_time})

        # print info
        print('--------------------------------------')
        print(f'Iteration:          {update}')
        print(f'Step:               {global_step}')
        print(f'Ep_num:             {sum_num}')
        print(f'EpRet_mean:         {np.mean(ep_ret_list):.2f}')
        print(f'EpRet_std:          {np.std(ep_ret_list):.2f}')
        print(f'EpLen_mean:         {np.mean(ep_len_list):.2f}')
        print(f'EpErr_mean:         {np.mean(ep_err_list):.2f}')
        print(f'EpErr_std:          {np.std(ep_err_list):.2f}')
        print(f'EpPsi_mean:         {np.mean(ep_psi_list):.2f}')
        print(f'EpPsi_std:          {np.std(ep_psi_list):.2f}')
        print(f'EpSuccess_num:      {success_num}')
        print(f'Value_loss:         {v_loss.item():.2f}')
        print(f'Policy_loss:        {pg_loss.item():.6f}')
        print(f'Old_approx_kl:      {old_approx_kl.item():.6f}')
        print(f'Approx_kl:          {approx_kl.item():.6f}')
        print(f'Clipfrac:           {np.mean(clipfracs):.6f}')
        print(f'Explained_variance: {explained_var:.6f}')
        print(f'Entropy:            {entropy_loss:.6f}')
        print(f'Time:               {str(datetime.timedelta(seconds=time.time() - start_time))}')
        print('--------------------------------------\n')

    file.close()
