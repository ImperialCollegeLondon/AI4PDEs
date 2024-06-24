#!/usr/bin/env python

# Copyright (c) 2024 
#
# Christopher C. Pain   c.pain@imperial.ac.uk
# Claire E. Heaney      c.heaney@imperial.ac.uk
# Boyang Chen           boyang.chen16@imperial.ac.uk
#
# Applied Modelling and Computation Group
# Department of Earth Science and Engineering
# Imperial College London
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#-- Import general libraries
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
    print('All the required 3D tensors have been created successfully!')
    print('===========================================================')
    print('values_u  => u velocity [first step]  - (1,1,nz,ny,nx)')
    print('values_v  => v velocity [first step]  - (1,1,nz,ny,nx)')
    print('values_w  => w velocity [first step]  - (1,1,nz,ny,nx)')
    print('values_p  => pressure                 - (1,1,nz,ny,nx)')
    print('b_uu      => v velocity [second step] - (1,1,nz+2,ny+2,nx+2)')
    print('b_vv      => v velocity [second step] - (1,1,nz+2,ny+2,nx+2)')
    print('b_ww      => w velocity [second step] - (1,1,nz+2,ny+2,nx+2)')
    print('values_uu => u velocity [first step]  - (1,1,nz+2,ny+2,nx+2)')
    print('values_vv => v velocity [first step]  - (1,1,nz+2,ny+2,nx+2)')
    print('values_ww => w velocity [first step]  - (1,1,nz+2,ny+2,nx+2)')
    print('values_pp => pressure                 - (1,1,nz+2,ny+2,nx+2)')
    print('===========================================================')
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
    print('All the required 2D tensors have been created successfully!')
    print('===========================================================')
    print('values_u  => u velocity [first step]  - (1,1,ny,nx)')
    print('values_v  => v velocity [first step]  - (1,1,ny,nx)')
    print('values_p  => pressure                 - (1,1,ny,nx)')
    print('b_uu      => v velocity [second step] - (1,1,ny+2,nx+2)')
    print('b_vv      => v velocity [second step] - (1,1,ny+2,nx+2)')
    print('values_uu => u velocity [first step]  - (1,1,ny+2,nx+2)')
    print('values_vv => v velocity [first step]  - (1,1,ny+2,nx+2)')
    print('values_pp => pressure                 - (1,1,ny+2,nx+2)')
    print('===========================================================')
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

def create_solid_body_2D(nx, ny, cor_x, cor_y, size_x, size_y):
    input_shape = (1, 1, ny, nx)
    sigma = torch.zeros(input_shape, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    sigma[0,0,cor_y-size_y:cor_y+size_y,cor_x-size_x:cor_x+size_x] = 1e08
    print('A bluff body has been created successfully!')
    print('===========================================')
    print('Size of body in x:',size_x*2)
    print('Size of body in y:',size_y*2)
    print('position of body in x:',cor_x)
    print('position of body in y:',cor_y)
    print('===========================================')
    return sigma