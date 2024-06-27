import torch

def boundary_condition_2D_u(u, ub, halo=1):
	u[0, 0,      :,  :halo].fill_(ub)	# inflow on the right x=0
	u[0, 0,      :, -halo:].fill_(ub)	# outflow on the left x=Lx
	u[0, 0,  :halo,      :].fill_(0.0)	# u=0 at y=0  (no-slip)
	u[0, 0, -halo:,      :].fill_(0.0)	# u=0 at y=Ly
	return u
    
def boundary_condition_2D_v(v, ub, halo=1):
	v[0, 0,      :,  :halo].fill_(0.0)	# v=0 at x=0 (no-slip)
	v[0, 0,      :, -halo:].fill_(0.0)	# v=0 at x=Lx (no-slip)
	v[0, 0,  :halo,      :].fill_(0.0)	# v=0 at y=0 (no inflow from south)
	v[0, 0, -halo:,      :].fill_(0.0)	# v=0 at y=Ly (no outflow to north)
	return v

def boundary_condition_2D_p(p):
	p[0, 0,  :,  0] = p[0, 0, :, 1]     # dp/dx=0 at x=0
	p[0, 0,  :, -1].fill_(0.0)		    # p=0 at x=Lx
	p[0, 0,  0,  :] = p[0, 0,  1, :]    # dp/dy=0 at y=0
	p[0, 0, -1,  :] = p[0, 0, -2, :]	# dp/dy=0 at y=Ly
	return p

# TODO generalise to halo>1
# def boundary_condition_2D_p(p, halo=1):
# 	halo_mirror = slice(2*halo-1, halo-1, -1)			# halo mirrored around x=y=0
# 	p[0, 0,      :,  :halo] = p[0, 0, :, halo_mirror]	# dp/dx=0 at x=0
# 	p[0, 0,      :, -halo:].fill_(0.0)					# p=0 at x=Lx
# 	p[0, 0,  :halo,      :] = p[0, 0,  halo_mirror, :]	# dp/dy=0 at y=0
	
# 	halo_mirror = slice(-halo-1, -2*halo-1, -1)			# halo mirrored around x=y=L
# 	p[0, 0, -halo:,  :] = p[0, 0, halo_mirror, :]		# dp/dy=0 at y=Ly
# 	return p

def boundary_condition_2D_cw(w):
	ny = w.shape[2]
	nx = w.shape[3]
	ww = torch.nn.functional.pad(w, (1, 1, 1, 1), mode='constant', value=0)
	ww[0, 0,    :,    0] = ww[0, 0,  :,  1]*0
	ww[0, 0,    :, nx+1] = ww[0, 0,  :, nx]*0
	ww[0, 0,    0,    :] = ww[0, 0,  1,  :]*0
	ww[0, 0, ny+1,    :] = ww[0, 0, ny,  :]*0
	return ww

def boundary_condition_3D_u(u, ub):
	# TODO generalise to halo > 1
	u[0, 0,  :,  :,  0].fill_(ub)	# inflow of ub at x=0
	u[0, 0,  :,  :, -1].fill_(ub)	# outflow of ub at x=lx
	u[0, 0,  :,  0,  :] = u[0, 0, :,  1, :]	# du/dy=0 at y=0
	u[0, 0,  :, -1,  :] = u[0, 0, :, -2, :]	# du/dy=0 at y=Ly
	u[0, 0,  0,  :,  :].fill_(0.0)	# u=0 at z=0 (no-slip at surface)
	u[0, 0, -1,  :,  :] = u[0, 0, -2, :, :]	# du/dz=0 at z=Lz (free-slip at top)
	return u
    
def boundary_condition_3D_v(v, ub):
	nz = values_v.shape[2]
	ny = values_v.shape[3]
	nx = values_v.shape[4]
	nnz = values_vv.shape[2]
	nny = values_vv.shape[3]
	nnx = values_vv.shape[4]

	values_vv[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_v[0,0,:,:,:]
	values_vv[0,0,:,:,0].fill_(0.0)
	values_vv[0,0,:,:,nx+1].fill_(0.0)
	values_vv[0,0,:,0,:].fill_(0.0)
	values_vv[0,0,:,ny+1,:].fill_(0.0)
	values_vv[0,0,0,:,:].fill_(0.0)
	values_vv[0,0,nz+1,:,:] = values_vv[0,0,nz,:,:]
	return values_vv

def boundary_condition_3D_w(values_w, values_ww, ub):
	nz = values_w.shape[2]
	ny = values_w.shape[3]
	nx = values_w.shape[4]
	nnz = values_ww.shape[2]
	nny = values_ww.shape[3]
	nnx = values_ww.shape[4]

	values_ww[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_w[0,0,:,:,:]
	values_ww[0,0,:,:,0].fill_(0.0)
	values_ww[0,0,:,:,nx+1].fill_(0.0)
	values_ww[0,0,:,0,:].fill_(0.0)
	values_ww[0,0,:,ny+1,:].fill_(0.0)
	values_ww[0,0,0,:,:].fill_(0.0)
	values_ww[0,0,nz+1,:,:] = values_ww[0,0,nz,:,:]
	return values_ww

def boundary_condition_3D_p(values_p, values_pp):
	nz = values_p.shape[2]
	ny = values_p.shape[3]
	nx = values_p.shape[4]
	nnz = values_pp.shape[2]
	nny = values_pp.shape[3]
	nnx = values_pp.shape[4]

	values_pp[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_p[0,0,:,:,:]
	values_pp[0,0,:,:,0] =  values_pp[0,0,:,:,1] 
	values_pp[0,0,:,:,nx+1].fill_(0.0)
	values_pp[0,0,:,0,:] = values_pp[0,0,:,1,:]
	values_pp[0,0,:,ny+1,:] = values_pp[0,0,:,ny,:]
	values_pp[0,0,0,:,:] = values_pp[0,0,1,:,:]
	values_pp[0,0,nz+1,:,:] = values_pp[0,0,nz,:,:]
	return values_pp

def boundary_condition_3D_k(k_u):
	k_uu = F.pad(k_u, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
	return k_uu

def boundary_condition_3D_cw(w):
	ww = F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
	return ww