import sys
import os
import numpy as np
from matplotlib import rcParams
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from IPython.display import Image
from IPython.core.display import HTML 


N_GRID_POINTS = 61
def plot_onedim(ax, gaussian, rotate=False):
    X_range = -5, 5
    X = np.linspace(X_range[0], X_range[1], 1001)[:, np.newaxis]
    pdf_vals = gaussian.pdf(X)
    if rotate:
        ax.plot(pdf_vals, X)
        ax.margins(y=0)

    else:
        ax.plot(X, pdf_vals)
        ax.margins(x=0)

    ax.set_title(gaussian.__str__(), fontsize=16)
    
def plot_twodim(ax, gaussian):
    N_GRID_POINTS = 61
    X_range = -5, 5
    Y_range = -5, 5
    X_plot_points = np.linspace(X_range[0], X_range[1], N_GRID_POINTS)
    Y_plot_points = np.linspace(Y_range[0], Y_range[1], N_GRID_POINTS)
    X, Y = np.meshgrid(X_plot_points, Y_plot_points)
    XY = np.append(X.reshape(-1, 1), Y.reshape(-1, 1), axis=1)

    pdf_vals = gaussian.pdf(XY.reshape(-1, 2)).reshape(N_GRID_POINTS, N_GRID_POINTS)

    ax.contourf(X, Y,pdf_vals)
    ax.set_title(gaussian.__str__(), fontsize=16)
    
def plot_gaussian(gaussian, condition_axis=None, condition_value=None, conditional_dist=None):
    
    if condition_axis is None:
        fig, ax = plt.subplots(1, figsize=(10, 6))
        if gaussian.dims == 1:
            plot_onedim(ax, gaussian)

        elif gaussian.dims == 2:
            plot_twodim(ax, gaussian)
            
        else:
            raise ValueError('Invalid number of dimensions, can only support one or two.')
        
    elif condition_axis == 0:
        fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = fig.add_gridspec(1, 3)
        ax0 = fig.add_subplot(gs[0, 0:2])
        ax1 = fig.add_subplot(gs[0, 2])
        plot_twodim(ax0, gaussian)
        ax0.axvline(condition_value, c='r')
        plot_onedim(ax1, conditional_dist, rotate=True)
        ax0.grid()
        ax1.grid()
        
    elif condition_axis == 1:
        fig = plt.figure(constrained_layout=True, figsize=(10, 9))
        gs = fig.add_gridspec(3, 1)
        ax0 = fig.add_subplot(gs[0:2])
        ax1 = fig.add_subplot(gs[2])
        plot_twodim(ax0, gaussian)
        ax0.axhline(condition_value, c='r')
        plot_onedim(ax1, conditional_dist)
        ax0.grid()
        ax1.grid()
        
    plt.show()


def show_marginals():

    # read images
    img_A = mpimg.imread(os.path.join(os.getcwd(), 'img/marg_y.png'))
    img_B = mpimg.imread(os.path.join(os.getcwd(), 'img/marg_x.png'))

    fig = plt.figure(constrained_layout=True, figsize=(16, 9))
    gs = fig.add_gridspec(1, 2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])
    ax0.set_title('Marginalizing C on y')
    ax1.set_title('Marginalizing D on x')
    ax0.imshow(img_A);
    ax1.imshow(img_B);

def show_images():

    # read images
    img_A = mpimg.imread(os.path.join(os.getcwd(), 'img/test_conditional_x.png'))
    img_B = mpimg.imread(os.path.join(os.getcwd(), 'img/test_conditional_y.png'))

    fig = plt.figure(constrained_layout=True, figsize=(16, 9))
    gs = fig.add_gridspec(1, 3)
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[0, 2])
    ax0.imshow(img_A);
    ax1.imshow(img_B);