import numpy as np
import matplotlib.pyplot as plt

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