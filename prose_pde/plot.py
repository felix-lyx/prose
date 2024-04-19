import seaborn as sns
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
import numpy as np
import os

def plot(u,utarget,types,p,notes = "",num_choice = 2,plot_dict = None, initial = False, plot_type = ["heat","advection","diff_react_1D","burgers","compressiveNS","twobody_diff_react_1D"]):
    try:
        os.makedirs("figures/" + p.exp_id)
    except FileExistsError:
        pass
    except OSError as error:
        print(f"Error: {error}")

    u_np = [useq.cpu().detach().numpy() for useq in u]
    nrows=1
    if utarget  is not None:
        utarget_np = [useq.cpu().detach().numpy() for useq in utarget]
        nrows = 2

    default_value = 0
    if plot_dict is None:
        num_plot = {key: default_value for key in plot_type}
    else:
        num_plot = plot_dict

    for i in range(len(u_np)):
        dim = u_np[i].shape[2]
        if types[i] in plot_type and num_plot[types[i]] < num_choice:
            num_plot[types[i]]+=1
            fig, ax = plt.subplots(nrows, dim, figsize=(12,6))
            ax = np.asarray([ax]).reshape((nrows,dim))


            for j in range(dim):
                vmin,vmax = np.min((np.asarray(u_np[i])[:, :, j])),np.max((np.asarray(u_np[i])[:, :]))
                if utarget is not None:
                    vmin,vmax = min(vmin,np.min((np.asarray(utarget_np[i])[:, :, j]))),max(vmin,np.max((np.asarray(utarget_np[i])[:, :, j])))
                sns.heatmap(np.asarray(u_np[i])[:, :, j], ax=ax[0,j], vmin=vmin, vmax=vmax)
                ax[0,j].set_xlabel('x')
                ax[0,j].set_ylabel('t')
                if utarget is not None:
                    ax[0,j].set_title('output')
                else:
                    ax[0, j].set_title('target')
                if utarget is not None:
                    sns.heatmap(np.asarray(utarget_np[i])[:, :, j], ax=ax[1,j], vmin=vmin, vmax=vmax)
                    ax[1,j].set_xlabel('x')
                    ax[1,j].set_ylabel('t')
                    ax[1,j].set_title('target')
            if utarget is not None:
                loss = "loss = {:.5f}".format(np.linalg.norm(np.asarray(u_np[i]) -np.asarray(utarget_np[i]))/np.linalg.norm(np.asarray(utarget_np[i])))
                title = types[i] + notes + loss

            else:
                title = types[i] + notes
            plt.suptitle(title)
            plt.savefig("figures/{}/{}_{}_eval_{}_{}.png".format(p.exp_id,types[i],p.validation_metrics,i,notes))
            plt.close(fig)
            if initial:
                initialcondition = u_np[i][0].flatten()
                plt.plot(initialcondition)
                plt.suptitle(title + "initial")
                plt.savefig("figures/{}/{}_{}_eval_{}_{}_initial.png".format(p.exp_id,types[i],p.validation_metrics,i,notes))
                plt.close()

    return num_plot


def plot_sample_output(u, utarget, x, t, p, title=""):
    try:
        os.makedirs("figures/" + p.exp_name + '/' + p.exp_id)
    except FileExistsError:
        pass
    except OSError as error:
        print(f"Error: {error}")

    unp = u.detach().numpy()
    utargetnp = utarget.detach().numpy()
    error = np.abs(unp - utargetnp)

    plt.style.use('default')  # Use a dark background for the plot
    # Setup the plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Define the number of ticks you want
    xticks = np.linspace(0, len(x) - 1, num=5, dtype=int)
    yticks = np.linspace(0, len(t) - 1, num=5, dtype=int)
    fig.patch.set_facecolor('white')  # Set the background color of the figure

    # Convert these tick locations to actual x and t values
    xticklabels = np.round(x[xticks].detach().numpy(), 2)
    yticklabels = np.round(t[yticks].detach().numpy(), 2)
    # Plot the exact solution
    vmin = np.min([utargetnp.min(), unp.min()])
    vmax = np.max([utargetnp.max(), unp.max()])
    sns.heatmap(utargetnp, ax=axs[0], cmap="jet", cbar=True,vmin = vmin, vmax = vmax)
    axs[0].set_title('Exact u(x, t)')
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xticklabels)
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels(yticklabels)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('t')
    axs[0].invert_yaxis()
    axs[0].set_facecolor('white')

    # Plot the predicted solution
    sns.heatmap(unp, ax=axs[1], cmap="jet", cbar=True, vmin = vmin, vmax = vmax)
    axs[1].set_title('Predicted u(x, t)')
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticklabels)
    axs[1].set_yticks(yticks)
    axs[1].set_yticklabels(yticklabels)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('t')
    axs[1].invert_yaxis()
    axs[1].set_facecolor('white')

    # Plot the absolute error
    sns.heatmap(error, ax=axs[2], cmap="jet", cbar=True)
    axs[2].set_title('Absolute error')
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(xticklabels)
    axs[2].set_yticks(yticks)
    axs[2].set_yticklabels(yticklabels)
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('t')
    axs[2].invert_yaxis()
    axs[2].set_facecolor('white')
    # Improve the layout
    plt.tight_layout()

    # Save the figure
    plt.gca().set_facecolor('none')
    plt.gca().set_alpha(0)
    fig.savefig("figures/{}/{}/{}.pdf".format(p.exp_name,p.exp_id, title), dpi=300,format='pdf', transparent=True)
    plt.close(fig)


def plot_sample_output_noerror(u, utarget, x, t, p, title=""):
    try:
        os.makedirs("figures/" + p.exp_name + '/' + p.exp_id)
    except FileExistsError:
        pass
    except OSError as error:
        print(f"Error: {error}")

    unp = u.detach().numpy()
    utargetnp = utarget.detach().numpy()

    plt.style.use('default')  # Use a dark background for the plot
    # Setup the plot
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.5))
    # Define the number of ticks you want
    xticks = np.linspace(0, len(x) - 1, num=5, dtype=int)
    yticks = np.linspace(1, len(t) - 1, num=3, dtype=int)
    fig.patch.set_facecolor('white')  # Set the background color of the figure

    # Convert these tick locations to actual x and t values
    xticklabels = np.round(x[xticks].detach().numpy(), 2)
    yticklabels = np.round(t[yticks].detach().numpy(), 2)
    # Plot the exact solution
    vmin = np.min([utargetnp.min(), unp.min()])
    vmax = np.max([utargetnp.max(), unp.max()])
    sns.heatmap(utargetnp, ax=axs[0], cmap="jet", cbar=True,vmin = vmin, vmax = vmax)
    axs[0].set_title('Exact u(x, t)')
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xticklabels)
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels(yticklabels)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('t')
    axs[0].invert_yaxis()
    axs[0].set_facecolor('white')

    # Plot the predicted solution
    sns.heatmap(unp, ax=axs[1], cmap="jet", cbar=True, vmin = vmin, vmax = vmax)
    axs[1].set_title('Predicted u(x, t)')
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticklabels)
    axs[1].set_yticks(yticks)
    axs[1].set_yticklabels(yticklabels)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('t')
    axs[1].invert_yaxis()
    axs[1].set_facecolor('white')

    # Improve the layout
    plt.tight_layout()

    # Save the figure
    plt.gca().set_facecolor('none')
    plt.gca().set_alpha(0)
    fig.savefig("figures/{}/{}/{}_noerror.pdf".format(p.exp_name,p.exp_id, title), dpi=300,format='pdf', transparent=True)
    plt.close(fig)





def plot_one(u, p, title="",input_size =5):
    try:
        os.makedirs("figures/" + p.exp_name + '/' + p.exp_id)
    except FileExistsError:
        pass
    except OSError as error:
        print(f"Error: {error}")

    unp = u.detach().numpy()


    # Setup the plot
    # plt.style.use('dark_background')  # Use a dark background for the plot
    fig, axs = plt.subplots(1, 1, figsize=(5, input_size))
    fig.patch.set_facecolor('white')  # Set the background color of the figure
    axs.set_facecolor('white')

    # Plot the predicted solution
    sns.heatmap(unp, ax=axs, cmap="jet", cbar=False)
    axs.set_xticks([])  # Remove xticks
    axs.set_yticks([])
    axs.invert_yaxis()


    # Improve the layout
    plt.tight_layout()

    # Save the figure
    plt.gca().set_facecolor('none')
    plt.gca().set_alpha(0)
    fig.savefig("figures/{}/{}/{}.pdf".format(p.exp_name,p.exp_id, title), dpi=300, facecolor=fig.get_facecolor(),format = 'pdf',transparent=True)
    plt.close(fig)
