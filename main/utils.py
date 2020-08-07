import numpy as np
import os


def p_law_burn(x, x_burn, expn, c0, cc):
    """Power law function with a burn-in period"""
    if x < x_burn:
        return c0
    else:
        return c0*cc / (cc + (x-x_burn)**expn)
    

def write_params(param_dict, dir_path, file_name):
    """Write a parameter file"""
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print ("Creation of the directory failed")
    f = open(dir_path + file_name, "w")
    for k,v in param_dict.items():
        f.write(k + "\t" + str(v) + "\n")
    f.close()
    

def read_params(path):
    """Read a parameter file"""
    params = dict()
    f = open(path, "r")
    for l in f.readlines():
        params[l.split()[0]] = float(l.split()[1])
    return params

    
def read_traj(path):
    """Read a trajectory with headers"""
    f = open(path, "r")
    v_traj = []
    state_labels = f.readline().split()
    for line in f.readlines():
        v_traj.append(line.split())
    return np.array(v_traj, dtype='float'), state_labels


def read_policy(path):
    """Read the policy trajectory"""
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
    return np.array(p_traj, dtype='float'), state_labels, action_labels


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



    
    
    