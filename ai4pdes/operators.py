#-- Import general libraries
import numpy as np
import torch

def get_weights_linear_2D(dx):
    w1 = torch.tensor([[[[1/3/dx**2], 
             [1/3/dx**2],
             [1/3/dx**2]],

            [[1/3/dx**2],
             [-8/3/dx**2],
             [1/3/dx**2]],

            [[1/3/dx**2],
             [1/3/dx**2],
             [1/3/dx**2]]]])

    w2 = torch.tensor([[[[1/(12*dx)],  # Central differencing for y-advection and second-order time scheme
             [0.0],
             [-1/(12*dx)]],

            [[1/(3*dx)],
             [0.0],
             [-1/(3*dx)]],

            [[1/(12*dx)],
             [0.0],
             [-1/(12*dx)]]]])

    w3 = torch.tensor([[[[-1/(12*dx)],  # Central differencing for y-advection and second-order time scheme
             [-1/(3*dx)],
             [-1/(12*dx)]],

            [[0.0],
             [0.0],
             [0.0]],

            [[1/(12*dx)],
             [1/(3*dx)],
             [1/(12*dx)]]]])

    wA = torch.tensor([[[[-1/3/dx**2],  # A matrix for Jacobi
             [-1/3/dx**2],
             [-1/3/dx**2]],

            [[-1/3/dx**2],
             [8/3/dx**2],
             [-1/3/dx**2]],

            [[-1/3/dx**2],
             [-1/3/dx**2],
             [-1/3/dx**2]]]])

    w1 = torch.reshape(w1, (1,1,3,3))
    w2 = torch.reshape(w2, (1,1,3,3))
    w3 = torch.reshape(w3, (1,1,3,3))
    wA = torch.reshape(wA, (1,1,3,3)) 
    w_res = torch.zeros([1,1,2,2]) 
    w_res[0,0,:,:] = 0.25
    diag = np.array(wA)[0,0,1,1]        # Diagonal component
    print('All the required 2D filters have been created successfully!')
    print('===========================================================')
    print('w1    => second order derivative  - (1,1,3,3)')
    print('w2    => first order derivative x - (1,1,3,3)')
    print('w3    => first order derivative y - (1,1,3,3)')
    print('wA    => second order derivative  - (1,1,3,3)')
    print('w_res => Restriction operation    - (1,1,3,3)')
    print('diag  => Diagonal component of wA - (1,1,1,1)')
    print('===========================================================')
    return w1, w2, w3, wA, w_res, diag

def get_weights_linear_3D(dx):
    pd1 = torch.tensor([[2/26, 3/26, 2/26],
           [3/26, 6/26, 3/26],
           [2/26, 3/26, 2/26]])
    pd2 = torch.tensor([[3/26, 6/26, 3/26],
           [6/26, -88/26, 6/26],
           [3/26, 6/26, 3/26]])
    pd3 = torch.tensor([[2/26, 3/26, 2/26],
           [3/26, 6/26, 3/26],
           [2/26, 3/26, 2/26]])
    w1 = torch.zeros([1, 1, 3, 3, 3])
    wA = torch.zeros([1, 1, 3, 3, 3])
    w1[0, 0, 0,:,:] = pd1/dx**2
    w1[0, 0, 1,:,:] = pd2/dx**2
    w1[0, 0, 2,:,:] = pd3/dx**2
    wA[0, 0, 0,:,:] = -pd1/dx**2
    wA[0, 0, 1,:,:] = -pd2/dx**2
    wA[0, 0, 2,:,:] = -pd3/dx**2
    # Gradient filters
    p_div_x1 = torch.tensor([[-0.014, 0.0, 0.014],
           [-0.056, 0.0, 0.056],
           [-0.014, 0.0, 0.014]])
    p_div_x2 = torch.tensor([[-0.056, 0.0, 0.056],
           [-0.22, 0.0, 0.22],
           [-0.056, 0.0, 0.056]])
    p_div_x3 = torch.tensor([[-0.014, 0.0, 0.014],
           [-0.056, 0.0, 0.056],
           [-0.014, 0.0, 0.014]])
    p_div_y1 = torch.tensor([[0.014, 0.056, 0.014],
           [0.0, 0.0, 0.0],
           [-0.014, -0.056, -0.014]])
    p_div_y2 = torch.tensor([[0.056, 0.22, 0.056],
           [0.0, 0.0, 0.0],
           [-0.056, -0.22, -0.056]])
    p_div_y3 = torch.tensor([[0.014, 0.056, 0.014],
           [0.0, 0.0, 0.0],
           [-0.014, -0.056, -0.014]])
    p_div_z1 = torch.tensor([[0.014, 0.056, 0.014],
           [0.056, 0.22, 0.056],
           [0.014, 0.056, 0.014]])
    p_div_z2 = torch.tensor([[0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0]])
    p_div_z3 = torch.tensor([[-0.014, -0.056, -0.014],
           [-0.056, -0.22, -0.056],
           [-0.014, -0.056, -0.014]])
    w2 = torch.zeros([1,1,3,3,3])
    w3 = torch.zeros([1,1,3,3,3])
    w4 = torch.zeros([1,1,3,3,3])
    w2[0,0,0,:,:] = -p_div_x1/dx*0.5
    w2[0,0,1,:,:] = -p_div_x2/dx*0.5
    w2[0,0,2,:,:] = -p_div_x3/dx*0.5
    w3[0,0,0,:,:] = -p_div_y1/dx*0.5
    w3[0,0,1,:,:] = -p_div_y2/dx*0.5
    w3[0,0,2,:,:] = -p_div_y3/dx*0.5
    w4[0,0,0,:,:] = -p_div_z1/dx*0.5
    w4[0,0,1,:,:] = -p_div_z2/dx*0.5
    w4[0,0,2,:,:] = -p_div_z3/dx*0.5
    # Restriction filters
    w_res = torch.zeros([1,1,2,2,2])
    w_res[0,0,:,:,:] = 0.125
    diag = np.array(wA)[0,0,1,1,1]    # Diagonal component
    print('All the required 3D filters have been created successfully!')
    print('===========================================================')
    print('w1    => second order derivative  - (1,1,3,3,3)')
    print('w2    => first order derivative x - (1,1,3,3,3)')
    print('w3    => first order derivative y - (1,1,3,3,3)')
    print('w4    => first order derivative z - (1,1,3,3,3)')
    print('wA    => second order derivative  - (1,1,3,3,3)')
    print('w_res => Restriction operation    - (1,1,3,3,3)')
    print('diag  => Diagonal component of wA - (1,1,1,1,1)')
    print('===========================================================')
    return w1, w2, w3, w4, wA, w_res, diag

