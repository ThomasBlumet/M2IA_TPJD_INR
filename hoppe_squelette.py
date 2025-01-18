
import numpy as np
import mcubes
from scipy import spatial

p = np.loadtxt('armadillo_sub.xyz')

#compute the enclosing grid
#TODO
maxx = np.max(p[:,0])
minx = np.min(p[:,0])
maxy = np.max(p[:,1])
miny = np.min(p[:,1])
maxz = np.max(p[:,2])
minz = np.min(p[:,2])

print(minx,maxx,miny,maxy,minz,maxz)
#normalize the shape
p[:,0] = 2 * (p[:,0] - minx)/(maxx - minx) - 1
p[:,1] = 2 * (p[:,1] - miny)/(maxy - miny) - 1
p[:,2] = 2 * (p[:,2] - minz)/(maxz - minz) - 1


grid_size = 10
X, Y, Z = np.mgrid[-1:1:10j, -1:1:10j, -1:1:10j]
kdtree = spatial.KDTree(p[:,:3])
u = np.zeros_like(X)

for i in range(1,10):
    for j in range(1,10):
        for k in range(1,10):
            query=[X[i,j,k], Y[i,j,k], Z[i,j,k]]
            distance,index = kdtree.query(query)
            normales = p[index,-3:]
            values = p[index,:3]
            vector= values-query
            #Extract the 0 level set (marching_cubes of the mcubes library)
            u[i,j,k] = np.dot(vector,normales)
            print(u[i,j,k])

#Extract the 0 level set (marching_cubes of the mcubes library)

vertices, triangles = mcubes.marching_cubes(u,0)

mcubes.export_obj(vertices, triangles, 'result.obj')

#show the tree mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(vertices[:,0], vertices[:,1], triangles, vertices[:,2], cmap='Spectral')
plt.show()