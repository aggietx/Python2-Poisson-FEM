import numpy as np
import scipy.sparse.linalg as spla
import cg
import matplotlib.pylab as plt	


plt.close('all')
print "FEM for poisson equation on uniform rectangle mesh, python 2 version "
## mesh
Lx=1.## domain size in x direction
Ly=1.## domain size in y direction
nx=128## number of elements in x direction
ny=64## number of elements in y direction
hx=Lx/nx## mesh size in x direction
hy=Ly/ny## mesh size in y direction
nnx=nx+1## number of points in x dirction
nny=ny+1## number of points in y dirction
n_nodes=nnx*nny##number of dof
n_elements=nx*ny## bumber of element
print "number of elements in x direction is %d "% nx
print "number of elements in y direction is %d "% ny
(bdnodes,left_nodes,right_nodes,bottom_nodes,top_nodes,free_nodes)=cg.getboundary_dof(nnx-1,nny-1);n_bdnodes=bdnodes.shape[0]
x = np.linspace(0, Lx, nnx)
y = np.linspace(0, Ly, nny)
xp, yp = np.meshgrid(x, y)
xp=np.reshape(xp,nnx*nny)
yp=np.reshape(yp,nnx*nny)
## coeff of the general elliptic operator, constant here
stiff_coeff=np.ones((nny-1,nnx-1))

## assemble global matrix and force
print "Assembling FEM matrix... "
local_massmatrix=cg.local_mass(hx,hy)## element stiff matrix
local_stiffmatrix=cg.local_stiff(hx,hy)## element mass matrix
Astiff=cg.assembleweightmatrix(stiff_coeff,local_stiffmatrix)
Massforce=cg.assembleweightmatrix(np.ones((nny-1,nnx-1)),local_massmatrix)
f=4*(-yp*yp+yp)*np.sin(np.pi*xp)
F=Massforce.dot(f)

print "Setting Dirichlet boundary condition..."

Astiff_free=Astiff[free_nodes][:,free_nodes]
F_free=F[free_nodes]

print "Solving linear system... "
u=np.zeros(n_nodes)
u[free_nodes]=spla.spsolve(Astiff_free, F_free)
#u[free_nodes]=spla.spsolve(Astiff_free, F_free,  use_umfpack=False)
print "computation is done"

## plot the solution
print "plot the solution"
um=np.reshape(u,(nny,nnx))
cg.plot_matrix(um,Lx,Ly)


