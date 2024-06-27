import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import matplotlib.animation as animation
from IPython.display import HTML

def plot_state(variable, variable_title):
    """Plot state"""
    fig = plt.figure(figsize=(15, 6))
    plt.imshow(-variable[0, 0, :, :])
    plt.colorbar()
    plt.title(variable_title)
    return fig

def plot_u(prognostic_variables):
    """Plot u component of velocity"""
    fig = plot_state(prognostic_variables.u.detach().numpy(), "u component of velocity (m/s)")
    return fig

def plot_v(prognostic_variables):
    """Plot v component of velocity"""
    fig = plot_state(prognostic_variables.v.detach().numpy(), "v component of velocity (m/s)")
    return fig

def plot_speed(prognostic_variables):
    """Plot speed"""
    fig = plot_state(np.sqrt(prognostic_variables.u.detach().numpy()**2 + 
                             prognostic_variables.v.detach().numpy()**2), "speed (m/s)")
    return fig


def plot_from_file(filename, variable_title):
    """Plot variable after saving to file. Filename should be .npy"""
    variable = np.load(filename)
    fig = plot_state(variable, variable_title)
    return fig

    

def plot_sensor(output):
    """Plot what the sensor sees"""
    fig = plt.figure(figsize=(15, 10))
    for i in range(output.nsensors):
        plt.subplot(output.nsensors, 1, i+1)
        plt.plot(output.sensor_locations[i,:],
                 label=f'Numerical probe {i+1}')
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('u velocity (m/s)')
    return fig

def animate_u(filepath, n_t, save_filename, dt=50):
    """Plot animation"""
    fig = plt.figure(figsize=(15, 10))
    u_t = np.load(f"{filepath}/u1.npy")
    im = plt.imshow(-u_t[0, 0, :, :]) #, vmin=-2, vmax=2)
    plt.colorbar()
    def update(t):
        # for each frame, update the data stored on each artist.
        # Open file for timestep t
        u_t = np.load(f"{filepath}/u{t*dt+1}.npy")
        im.set_array(-u_t[0, 0, :, :])
        plt.title(f"Flow at time {t*dt+1}")
 
    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_t-1,
                                  interval=30, blit=False)
    writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
    ani.save(save_filename, writer=writer)
    print(f"Animation generated and saved as {save_filename}.")

