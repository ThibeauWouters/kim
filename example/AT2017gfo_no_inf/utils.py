import os
import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import jax.numpy as jnp
import corner
import jax
from jaxtyping import Array, Float

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False,
                        truth_color="red")

matplotlib_params = {"axes.grid": True,
        "text.usetex" : False,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(matplotlib_params)

labels = ["log10_mej_dyn", "vej_dyn", "Yedyn", "log10_mej_wind", "vej_wind", "inclination_EM"]
n_dim = len(labels)

################
### PLOTTING ###
################

def plot_accs(accs, label, name, outdir):
    
    eps = 1e-3
    plt.figure(figsize=(10, 6))
    plt.plot(accs, label=label)
    plt.ylim(0 - eps, 1 + eps)
    
    plt.ylabel(label)
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()
    
def plot_log_prob(log_prob, label, name, outdir):
    log_prob = np.mean(log_prob, axis = 0)
    plt.figure(figsize=(10, 6))
    plt.plot(log_prob, label=label)
    # plt.yscale('log')
    plt.ylabel(label)
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()

    
def plot_chains(chains, name, outdir, labels = labels):
    
    chains = np.array(chains)
    
    # Check if 3D, then reshape
    if len(np.shape(chains)) == 3:
        chains = chains.reshape(-1, n_dim)
    
    chains = np.asarray(chains)
    fig = corner.corner(chains, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
    fig.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    
def plot_chains_from_file(outdir):
    
    filename = outdir + 'results_production.npz'
    data = np.load(filename)
    chains = data['chains']
    my_chains = []
    n_dim = np.shape(chains)[-1]
    for i in range(n_dim):
        values = chains[:, :, i].flatten()
        my_chains.append(values)
    my_chains = np.array(my_chains).T
    chains = chains.reshape(-1, n_dim)
    
    plot_chains(chains, 'results', outdir)
    
def plot_accs_from_file(outdir):
    
    filename = outdir + 'results_production.npz'
    data = np.load(filename)
    local_accs = data['local_accs']
    global_accs = data['global_accs']
    
    local_accs = np.mean(local_accs, axis = 0)
    global_accs = np.mean(global_accs, axis = 0)
    
    plot_accs(local_accs, 'local_accs', 'local_accs_production', outdir)
    plot_accs(global_accs, 'global_accs', 'global_accs_production', outdir)
    
def plot_log_prob_from_file(outdir, which_list = ['training', 'production']):
    
    for which in which_list:
        filename = outdir + f'results_{which}.npz'
        data = np.load(filename)
        log_prob= data['log_prob']
        plot_log_prob(log_prob, f'log_prob_{which}', f'log_prob_{which}', outdir)
    
def plot_loss_vals(loss_values, label, name, outdir):
    loss_values = loss_values.reshape(-1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label=label)
    
    plt.ylabel(label)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()

