import torch
import torch.nn as nn
import torch.nn.functional as F

def boundary_condition_Dirichlet_2D(values, values_pad, face):
	'''
	face
	0 - west 0
	1 - east nx
	2 - top  0
	3 - bot  ny
	'''
	ny = values.shape[2]
	nx = values.shape[3]
	nny = values_pad.shape[2]
	nnx = values_pad.shape[3]
	if face == 0:
		values_pad[0,0,:,0].fill_(0.0)
	elif face == 1:
		values_pad[0,0,:,nx+1].fill_(0.0)
	elif face == 2:
		values_pad[0,0,0,:].fill_(0.0)
	elif face == 3:
		values_pad[0,0,ny+1,:].fill_(0.0)
	return values, values_pad

def boundary_condition_Neumann_2D(values, values_pad, face):
	'''
	face
	0 - west 0
	1 - east nx
	2 - top  0
	3 - bot  ny
	'''
	ny = values.shape[2]
	nx = values.shape[3]
	nny = values_pad.shape[2]
	nnx = values_pad.shape[3]
	if face == 0:
		values_pad[0,0,:,0] = values_pad[0,0,:,1]
	elif face == 1:
		values_pad[0,0,:,nx+1] = values_pad[0,0,:,nx]
	elif face == 2:
		values_pad[0,0,0,:] = values_pad[0,0,1,:]
	elif face == 3:
		values_pad[0,0,ny+1,:] = values_pad[0,0,ny,:]
	return values, values_pad

def boundary_condition_inflow_2D(values, values_pad, face, ub):
	'''
	face
	0 - west 0
	1 - east nx
	2 - top  0
	3 - bot  ny
	'''
	ny = values.shape[2]
	nx = values.shape[3]
	nny = values_pad.shape[2]
	nnx = values_pad.shape[3]
	if face == 0:
		values_pad[0,0,:,0].fill_(ub)
	elif face == 1:
		values_pad[0,0,:,nx+1].fill_(ub)
	elif face == 2:
		values_pad[0,0,0,:].fill_(ub)
	elif face == 3:
		values_pad[0,0,ny+1,:].fill_(ub)
	return values, values_pad

def boundary_condition_outflow_2D(values, values_pad, face, ub):
	'''
	face
	0 - west 0
	1 - east nx
	2 - top  0
	3 - bot  ny
	'''
	ny = values.shape[2]
	nx = values.shape[3]
	nny = values_pad.shape[2]
	nnx = values_pad.shape[3]
	if face == 0:
		values_pad[0,0,:,0].fill_(ub)
	elif face == 1:
		values_pad[0,0,:,nx+1].fill_(ub)
	elif face == 2:
		values_pad[0,0,0,:].fill_(ub)
	elif face == 3:
		values_pad[0,0,ny+1,:].fill_(ub)
	return values, values_pad

def boundary_condition_u(values_u, values_uu,ub):
	ny = values_u.shape[2]
	nx = values_u.shape[3]
	nny = values_uu.shape[2]
	nnx = values_uu.shape[3]

	values_uu[0,0,1:nny-1,1:nnx-1] = values_u[0,0,:,:]
	values_uu[0,0,:,0].fill_(ub)
	values_uu[0,0,:,nx+1].fill_(ub)
	values_uu[0,0,0,:].fill_(0.0)
	values_uu[0,0,ny+1,:].fill_(0.0)
	return values_uu
    
def boundary_condition_v(values_v, values_vv,ub):
	ny = values_v.shape[2]
	nx = values_v.shape[3]
	nny = values_vv.shape[2]
	nnx = values_vv.shape[3]

	values_vv[0,0,1:nny-1,1:nnx-1] = values_v[0,0,:,:]
	values_vv[0,0,:,0].fill_(0.0)
	values_vv[0,0,:,nx+1].fill_(0.0)
	values_vv[0,0,0,:].fill_(0.0)
	values_vv[0,0,ny+1,:].fill_(0.0)
	return values_vv

def boundary_condition_p(values_p, values_pp):
	ny = values_p.shape[2]
	nx = values_p.shape[3]
	nny = values_pp.shape[2]
	nnx = values_pp.shape[3]

	values_pp[0,0,1:nny-1,1:nnx-1] = values_p[0,0,:,:]
	values_pp[0,0,:,0] =  values_pp[0,0,:,1] 
	values_pp[0,0,:,nx+1] = values_pp[0,0,:,nx]*0
	values_pp[0,0,0,:] = values_pp[0,0,1,:]
	values_pp[0,0,ny+1,:] = values_pp[0,0,ny,:]
	return values_pp

def boundary_condition_cw(w):
	ny = w.shape[2]
	nx = w.shape[3]
	ww = F.pad(w, (1, 1, 1, 1), mode='constant', value=0)
	ww[0,0,:,0] =  ww[0,0,:,1]*0
	ww[0,0,:,nx+1] = ww[0,0,:,nx]*0
	ww[0,0,0,:] = ww[0,0,1,:]*0
	ww[0,0,ny+1,:] = ww[0,0,ny,:]*0
	return ww