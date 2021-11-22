import numpy as np
import matplotlib.pyplot as plt
import os


def p_law_burn(x, x_burn, expn, c0, cc):
    """Power law function with a burn-in period"""
    if x < x_burn:
        return c0
    else:
        return c0*cc / (cc + (x-x_burn)**expn)

def p_lin_burn(x, x_burn, expn, c0, cc):
    """Linearly-decreasing function with a burn-in period"""
    if x < x_burn:
        return c0
    else:
        return c0 - (cc * (x-x_burn)**expn)

def p_mix_burn(x, x_burn0, x_burn1, expn, c0, c1, cc):
    """Linearly-decreasing and then power law function with a burn-in period"""
    if x < x_burn0:
        return c0
    else:
        if (x > x_burn0) and (x < x_burn1):
            return c0 - ((c0 - c1)/(x_burn1 - x_burn0))*(x-x_burn0)
        else:
            return c1*cc / (cc + (x-x_burn1)**expn)

def p_law_state(n, n0, expn, c0):
    """Power-law decreasing function depending on the occupation number"""
    return c0 / (1 + (n/n0)**expn)

def p_2step_burn(x, x_burn, c0, c1):
    """Two-step function with two burn-in period"""
    if x < x_burn:
        return c0
    else:
        return c1


def plot_lr_eps(alg_params):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,3))

    ax1.set_xlabel('Learning step', fontsize=14)
    ax1.set_ylabel('Learning rate', fontsize=14)
    ax1.set_yscale('log')
    xs = np.linspace(0, alg_params['n_steps'], 100)
    lr = [p_law_burn(x, alg_params['lr_burn'], alg_params['lr_expn'], alg_params['lr0'], alg_params['lrc']) for x in xs]
    ax1.plot(xs, lr)
    ax2.set_xlabel('Learning step', fontsize=14)
    ax2.set_ylabel('Epsilon', fontsize=14)
    ax2.set_yscale('log')
    eps = [p_law_burn(x, alg_params['eps_burn'], alg_params['eps_expn'], alg_params['eps0'], alg_params['epsc']) for x in xs]
    ax2.plot(xs, eps)
    return fig, (ax1, ax2)

def plot_lr_eps2(alg_params):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,3))

    ax1.set_xlabel('Learning step', fontsize=14)
    ax1.set_ylabel('Learning rate', fontsize=14)
    ax1.set_yscale('log')
    xs = np.linspace(0, alg_params['n_steps'], 100)
    lr = [p_mix_burn(x, alg_params['lr_burn0'], alg_params['lr_burn1'], alg_params['lr_expn'], alg_params['lr0'], alg_params['lr1'], alg_params['lrc']) for x in xs]
    ax1.plot(xs, lr)
    ax2.set_xlabel('Learning step', fontsize=14)
    ax2.set_ylabel('Epsilon', fontsize=14)
    ax2.set_yscale('log')
    eps = [p_law_burn(x, alg_params['eps_burn'], alg_params['eps_expn'], alg_params['eps0'], alg_params['epsc']) for x in xs]
    ax2.plot(xs, eps)
    return fig, (ax1, ax2)

def plot_lr_eps3(alg_params):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,3))

    ax1.set_xlabel('Learning step', fontsize=14)
    ax1.set_ylabel('Learning rate', fontsize=14)
    ax1.set_yscale('log')
    xs = np.linspace(0, alg_params['n_steps'], 100)
    lr = [p_2step_burn(x, alg_params['lr_burn0'], alg_params['lr0'], alg_params['lr1']) for x in xs]
    ax1.plot(xs, lr)
    ax2.set_xlabel('Learning step', fontsize=14)
    ax2.set_ylabel('Epsilon', fontsize=14)
    ax2.set_yscale('log')
    eps = [p_law_burn(x, alg_params['eps_burn'], alg_params['eps_expn'], alg_params['eps0'], alg_params['epsc']) for x in xs]
    ax2.plot(xs, eps)
    return fig, (ax1, ax2)

def plot_lr_eps_state(alg_params):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,3))

    ax1.set_xlabel('Learning step', fontsize=14)
    ax1.set_ylabel('Learning rate', fontsize=14)
    ax1.set_yscale('log')
    xs = np.linspace(0, alg_params['n_steps'], 100)
    lr = [p_law_state(x, alg_params['n0'], alg_params['lr_expn'], alg_params['lr0']) for x in xs]
    ax1.plot(xs, lr)
    ax2.set_xlabel('Learning step', fontsize=14)
    ax2.set_ylabel('Epsilon', fontsize=14)
    ax2.set_yscale('log')
    eps = [p_law_burn(x, alg_params['eps_burn'], alg_params['eps_expn'], alg_params['eps0'], alg_params['epsc']) for x in xs]
    ax2.plot(xs, eps)
    return fig, (ax1, ax2)

def plot_lr_eps_2state(alg_params):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,3))

    ax1.set_xlabel('Number of visits', fontsize=14)
    ax1.set_ylabel('Learning rate', fontsize=14)
    ax1.set_yscale('log')
    xs = np.linspace(0, alg_params['n_steps'], 100)
    lr = [p_law_state(x, alg_params['n0'], alg_params['lr_expn'], alg_params['lr0']) for x in xs]
    ax1.plot(xs, lr)
    ax2.set_xlabel('Learning step', fontsize=14)
    ax2.set_ylabel('Epsilon', fontsize=14)
    ax2.set_yscale('log')
    eps = [p_law_burn(x, alg_params['eps_burn'], alg_params['eps_expn'], alg_params['eps0'], alg_params['epsc']) for x in xs]
    ax2.plot(xs, eps)
    return fig, (ax1, ax2)



def write_params(param_dict, dir_path, file_name):
    """Write a parameter file"""
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print ("Creation of the directory failed")
    f = open(dir_path + file_name, "w")
    for k,v in param_dict.items():
        if type(v) is list or type(v) is np.ndarray:
            f.write(k + "\t")
            for i in range(len(v)):
                f.write(str(v[i])+",")
            f.write("\n")
        else:
            f.write(k + "\t" + str(v) + "\n")
    f.close()


def read_params(path):
    """Read a parameter file"""
    params = dict()
    f = open(path, "r")
    for l in f.readlines():
        try:
            params[l.split()[0]] = float(l.split()[1])
        except ValueError:
            if ',' not in l.split()[1]:
                params[l.split()[0]] = l.split()[1]
            else:
                params[l.split()[0]] = np.array(l.split()[1].split(',')[:-1], dtype=float)
    return params


def read_traj(path):
    """Read a trajectory with headers"""
    f = open(path, "r")
    v_traj = []
    state_labels = f.readline().split()
    for line in f.readlines():
        v_traj.append(line.split())
    try:
        return np.array(v_traj, dtype='float'), state_labels
    except ValueError:
        return np.array(v_traj), state_labels


def read_2d_traj(path):
    """Read a two dimensional trajectory"""
    f = open(path, "r")
    state_labels = f.readline().split()
    action_labels = f.readline().split(',')
    p_traj = []
    for line in f.readlines():
        policy_at_time = []
        for elem in line.split():
            policy_at_state = np.array(elem.split(','), dtype=float)
            policy_at_time.append(policy_at_state)
        p_traj.append(policy_at_time)
    #return np.array(p_traj, dtype='float'), state_labels, action_labels
    return p_traj, state_labels, action_labels

def read_best_quality(path):
    f = open(path, "r")
    b = []
    for line in f.readlines():
        for elem in line.split():
            quality_at_state = np.array(elem.split(','), dtype=float)
            b.append(quality_at_state)
    return b

def read_best_policy(path):
    f = open(path, "r")
    b = []
    for line in f.readlines():
        policy_at_state = np.array([int(x) for x in line.split()])
        b.append(policy_at_state)
    return b

def smooth_traj(traj, wind_size):
    """Binned average over the trajectory, with bin of size wind_size.
    It returns the binned x-axis and the averaged y-axis"""
    new_traj, times = [], []
    i = 0
    while i < len(traj)-wind_size:
        new_traj.append(np.mean(traj[i:i+wind_size]))
        times.append(i + wind_size/2)
        i += wind_size
    return times, new_traj

def read_occ(path):
    f = open(path, "r")
    n = []
    for line in f.readlines():
        for elem in line.split():
            n_s = np.array(elem.split(','), dtype=int)
        n_sum = np.sum(n_s)
        n.append(n_sum)
    return n
