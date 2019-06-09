import scipy.sparse as sparse
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib.pylab as plt
import numpy.matlib

def local_stiff(hx,hy): ## element stiffness matrix
	local_matrix=np.zeros((4,4))
	local_matrix[0,0]=(hx*hx+hy*hy)/(3*hx*hy)
	local_matrix[0,1]=hx/(6*hy)-hy/(3*hx)
	local_matrix[0,2]=-(hx*hx+hy*hy)/(6*hx*hy)
	local_matrix[0,3]=-hx/(3*hy)+hy/(6*hx)

	local_matrix[1,0]=local_matrix[0,1]
	local_matrix[1,1]=local_matrix[0,0]
	local_matrix[1,2]=local_matrix[0,3]
	local_matrix[1,3]=local_matrix[0,2]

	local_matrix[2,0]=local_matrix[0,2]
	local_matrix[2,1]=local_matrix[1,2]
	local_matrix[2,2]=local_matrix[0,0]
	local_matrix[2,3]=local_matrix[0,1]

	local_matrix[3,0]=local_matrix[0,3]
	local_matrix[3,1]=local_matrix[1,3]
	local_matrix[3,2]=local_matrix[2,3]
	local_matrix[3,3]=local_matrix[0,0]
	return local_matrix

def local_mass(hx,hy): ## element mass matrix
	local_matrix=np.zeros((4,4))
	local_matrix[0,0]=1./9
	local_matrix[0,1]=1./18
	local_matrix[0,2]=1./36
	local_matrix[0,3]=1./18

	local_matrix[1,0]=local_matrix[0,1]
	local_matrix[1,1]=local_matrix[0,0]
	local_matrix[1,2]=local_matrix[0,3]
	local_matrix[1,3]=local_matrix[0,2]

	local_matrix[2,0]=local_matrix[0,2]
	local_matrix[2,1]=local_matrix[1,2]
	local_matrix[2,2]=local_matrix[0,0]
	local_matrix[2,3]=local_matrix[0,1]

	local_matrix[3,0]=local_matrix[0,3]
	local_matrix[3,1]=local_matrix[1,3]
	local_matrix[3,2]=local_matrix[2,3]
	local_matrix[3,3]=local_matrix[0,0]
	local_matrix=hx*hy*local_matrix
	return local_matrix
	
def localdof_2globaldof(ny,nx): ## local dof in each element in global dof
	ngrid=nx*ny

	total_nodes=(nx+1)*(ny+1)
	all_nodes_vec=np.arange(0,total_nodes)
	all_nodes=np.reshape(all_nodes_vec,(ny+1,nx+1))
	nod=np.zeros((4,ngrid),dtype=np.int)
	sx=all_nodes[0:ny,0:nx];first_node=np.reshape(sx,nx*ny)
	nod[0,:]=first_node
	nod[1,:]=first_node+1
	nod[2,:]=first_node+nx+2
	nod[3,:]=first_node+nx+1
	
	gridx=np.matlib.repmat(nod, 4, 1)
	gridy=np.zeros((16,ngrid),dtype=np.int)
	gridy[0:4,:]=np.matlib.repmat(first_node, 4, 1)
	gridy[4:8,:]=np.matlib.repmat(first_node+1, 4, 1)
	gridy[8:12,:]=np.matlib.repmat(first_node+nx+2, 4, 1)
	gridy[12:16,:]=np.matlib.repmat(first_node+nx+1, 4, 1)
	return (gridx,gridy)

def assembleweightmatrix(coeff,local_matrix):## assemble global matrix with element matrix and coeff available
	nx=coeff.shape[1]
	ny=coeff.shape[0]
	ngrid=nx*ny
	(gridx,gridy)=localdof_2globaldof(ny,nx)
	total_nodes=(nx+1)*(ny+1)


	coeffvec=np.reshape(coeff,(1,nx*ny))#print k
	lk=np.reshape(local_matrix,(16,1))
	tempk=lk.dot(coeffvec)

	gdx=np.reshape(gridx,ngrid*16)
	gdy=np.reshape(gridy,ngrid*16)
	gdvalue=np.reshape(tempk,ngrid*16)

	
	globalmatrix=sparse.csr_matrix((gdvalue, (gdy, gdx)), shape=(total_nodes,total_nodes))
	#globalmatrix=sparse.lil_matrix(globalmatrix)
	return globalmatrix


def getboundary_dof(nx,ny):
	total_nodes=(nx+1)*(ny+1)
	all_nodes_vec=np.arange(0,total_nodes)
	all_nodes=np.reshape(all_nodes_vec,(ny+1,nx+1))
	left_nodes=all_nodes[:,0]
	right_nodes=all_nodes[:,nx]
	bottom_nodes=all_nodes[0,:]
	top_nodes=all_nodes[ny,:]
	free_nodes=all_nodes[1:ny,1:nx];free_nodes=np.reshape(free_nodes,(ny-1,nx-1))
	free_nodes=np.reshape(free_nodes,(nx-1)*(ny-1))
	boundary_nodes=np.setdiff1d(all_nodes_vec,free_nodes)
	return (boundary_nodes,left_nodes,right_nodes,bottom_nodes,top_nodes,free_nodes)


def plot_matrix(matrix,Lx,Ly):
	#fig = plt.figure()
	#ax =plt.subplots
	plt.imshow(matrix, extent=[0,Lx,0,Ly])
	plt.colorbar()
	plt.draw()
	plt.show(block=False)
def plot_vector(u,nx,ny):
## assume size of u is nx*ny
	u=np.reshape(u,(ny,nx))
	plot_matrix(u,1,1)
def regularization(regular,A):
	A=A+regular*sparse.dia_matrix((A.diagonal(), 0),shape=(A.shape[0], A.shape[1]))
	return A
