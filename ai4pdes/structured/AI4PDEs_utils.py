import numpy as np
import torch

def create_tensors_3D(nx, ny, nz):
    input_shape = (1, 1, nz, ny, nx)
    input_shape_pad = (1, 1, nz + 2, ny + 2, nx + 2)
    values_u = torch.zeros(input_shape, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_v = torch.zeros(input_shape, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_w = torch.zeros(input_shape, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_p = torch.zeros(input_shape, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_uu = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_vv = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_ww = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_pp = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    b_uu = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    b_vv = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    b_ww = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return values_u, values_v, values_w, values_p, values_uu, values_vv, values_ww, values_pp, b_uu, b_vv, b_ww

def create_tensors_2D(nx, ny):
    input_shape = (1, 1, ny, nx)
    input_shape_pad = (1, 1, ny + 2, nx + 2)
    values_u = torch.zeros(input_shape, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_v = torch.zeros(input_shape, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_p = torch.zeros(input_shape, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_uu = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_vv = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    values_pp = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    b_uu = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    b_vv = torch.zeros(input_shape_pad, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print('All the required tensors have been created successfully!')
    return values_u, values_v, values_p, values_uu, values_vv, values_pp, b_uu, b_vv

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
    print('All the required filters have been created successfully!')
    return w1, w2, w3, wA, w_res, diag

def create_solid_body_2D(nx, ny, cor_x, cor_y, size_x, size_y):
    input_shape = (1, 1, ny, nx)
    sigma = torch.zeros(input_shape, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    sigma[0,0,cor_y-size_y:cor_y+size_y,cor_x-size_x:cor_x+size_x] = 1e08
    print('A bluff body has been created successfully!')
    return sigma
