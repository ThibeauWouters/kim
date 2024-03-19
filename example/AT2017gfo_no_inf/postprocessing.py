import numpy as np
import matplotlib.pyplot as plt
import corner
import pickle

params = {"axes.grid": True,
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

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
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
                        save=False)

# import utils

# Load the data
filename = "./outdir/samples.pkl"
naming = ["log10_mej_dyn", "vej_dyn", "Yedyn", "log10_mej_wind", "vej_wind", "inclination_EM"]
with open(filename, "rb") as f:
    samples = pickle.load(f)
    
print(samples)
print(type(samples))

values = np.array([samples[key].flatten() for key in naming]).T
print(np.shape(values))

corner.corner(samples, labels = naming, **default_corner_kwargs)
plt.savefig("./outdir/test_corner.png", bbox_inches = "tight")
plt.close()