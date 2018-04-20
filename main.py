import sys
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from readInp import readInpFile
from Element2D import *

if len(sys.argv) == 1:
    filename = 'test.inp'
else:
    filename = sys.argv[1]

nodes, nodeSets, elements, elementSets = readInpFile(filename)
nodes = nodes[:,:2]
print(len(nodes), "nodes")
print(len(elements), "elements")

def showShape():
    plt.subplot(121).set_aspect('equal')
    plt.scatter(nodes[:,0], nodes[:,1])
    for k in list(nodeSets)[::-1]:
        nodeSet = nodes[nodeSets[k]]
        plt.scatter(nodeSet[:,0], nodeSet[:,1])
    plt.legend(['all'] + list(nodeSets)[::-1], loc='best')
    for i, p in enumerate(nodes):
        plt.text(p[0], p[1], r'$\leftarrow$'+str(i))

    plt.subplot(122).set_aspect('equal')
    for e in elements:
        nL = nodes[e[1]]
        plt.plot(nL[:,0].T, nL[:,1].T)
    plt.show()

nN = len(nodes)

if nN < 100:
    showShape()

print('constructing linear system...')
# construct linear system
val = []
row = []
col = []
rhs = np.zeros(nN * 2)

print('constructing matrix...')
# Material
tags = list(elementSets)
for t in tags:
    if t.startswith("Material"):
        para = {k: float(v)
                for c in t.split(':')[1:]
                for k,v in (c.split('='),)
                }
        print(t, para)
        for el in elementSets[t]:
            tp, ns = elements[el]
            if tp == 'CPS4' and len(ns) == 9:
                subMat = CPS9StiffMat(nodes[ns], para['E'], para['nu'])
            elif tp == 'CPS8' and len(ns) == 8:
                subMat = CPS8StiffMat(nodes[ns], para['E'], para['nu'])
            elif tp == 'CPS4' and len(ns) == 4:
                subMat = CPS4StiffMat(nodes[ns], para['E'], para['nu'])
            elif tp == 'CPS6' and len(ns) == 6:
                subMat = CPS6StiffMat(nodes[ns], para['E'], para['nu'])
            elif tp == 'CPS3' and len(ns) == 3:
                subMat = CPS3StiffMat(nodes[ns], para['E'], para['nu'])
            else:
                raise Exception('invalid element {} with {} nodes'.format(tp, len(ns)))
            indices = [i for n in ns for i in (2 * n, 2 * n + 1)]
            c, r = np.meshgrid(indices, indices)
            col += c.flatten().tolist()
            row += r.flatten().tolist()
            val += subMat.flatten().tolist()
mat = sparse.coo_matrix((val, (row, col)), shape=(nN * 2, nN * 2))

print('constructing rhs...')
# Load
for t in tags:
    if t.startswith("Load"):
        para = {k: float(v)
                for c in t.split(':')[1:]
                for k,v in (c.split('='),)
                }
        print(t, para)
        for el in elementSets[t]:
            tp, ns = elements[el]
            if tp == 'T3D3' and len(ns) == 3:
                subRhs = T3D3Force(nodes[ns], para['P'])
            elif tp == 'T3D2' and len(ns) == 2:
                subRhs = T3D2Force(nodes[ns], para['P'])
            elif tp == 'Node' and len(ns) == 1:
                F_x = para['F_x'] if 'F_x' in para else 0
                F_y = para['F_y'] if 'F_y' in para else 0
                subRhs = np.array((F_x, F_y))
            else:
                raise Exception('invalid element {} with {} nodes'.format(tp, len(ns)))
            indices = [i for n in ns for i in (2 * n, 2 * n + 1)]
            rhs[indices] += subRhs

print('applying constraints...')
# Constraint
indices = []
for t in tags:
    if t.startswith("Constraint"):
        para = {k: float(v)
                for c in t.split(':')[1:]
                for k,v in (c.split('='),)
                }
        print(t, para)
        for el in elementSets[t]:
            tp, ns = elements[el]
            #assert tp == 'T3D3' and len(ns) == 3
            if 'u' in para:
                assert para['u'] == 0
                for ni in ns:
                    indices.append(ni * 2)
            if 'v' in para:
                assert para['v'] == 0
                for ni in ns:
                    indices.append(ni * 2 + 1)
indices = list(set(indices))
indices.sort()

# TODO which is better?
tmax = mat.max()
mat = mat.tolil()
mat[indices, :] = 0
mat[:, indices] = 0
for i in indices:
    mat[i, i] = tmax
mat = mat.tocsr()

# mat = mat.tocsr()
# for i in indices:
#     mat[i, i] += 1e10

rhs[indices] = 0

def showLoad():
    plt.subplot(121).set_aspect('equal')
    plt.title("F_x")
    plt.scatter(nodes[:,0],nodes[:,1],c=rhs[::2])
    plt.colorbar()
    plt.subplot(122).set_aspect('equal')
    plt.title("F_y")
    plt.scatter(nodes[:,0],nodes[:,1],c=rhs[1::2])
    plt.colorbar()
    plt.show()

showLoad()

print("solving linear system")
import scipy.sparse.linalg

# spsolve is more stable and fast than cg/cgs
disp = sparse.linalg.spsolve(mat, rhs)

disp.shape = (nN, 2)

print('calc nodal stress')
# Material
tags = list(elementSets)
stress = np.zeros((disp.shape[0], 3))
count = np.zeros(disp.shape[0], dtype=np.int)
for t in tags:
    if t.startswith("Material"):
        para = {k: float(v)
                for c in t.split(':')[1:]
                for k,v in (c.split('='),)
                }
        print(t, para)
        for el in elementSets[t]:
            tp, ns = elements[el]
            if tp == 'CPS4' and len(ns) == 9:
                subStress = CPS9NodalStress(nodes[ns], para['E'], para['nu'], disp[ns])
            elif tp == 'CPS8' and len(ns) == 8:
                subStress = CPS8NodalStress(nodes[ns], para['E'], para['nu'], disp[ns])
            elif tp == 'CPS4' and len(ns) == 4:
                subStress = CPS4NodalStress(nodes[ns], para['E'], para['nu'], disp[ns])
            elif tp == 'CPS6' and len(ns) == 6:
                subStress = CPS6NodalStress(nodes[ns], para['E'], para['nu'], disp[ns])
            elif tp == 'CPS3' and len(ns) == 3:
                subStress = CPS3NodalStress(nodes[ns], para['E'], para['nu'], disp[ns])
            else:
                raise Exception('invalid element {} with {} nodes'.format(tp, len(ns)))
            ns = ns[:len(subStress)]
            stress[ns] += subStress
            count[ns] += 1

stress /= np.reshape(count, (count.size, 1))

# show displacement
plt.subplot(131).set_aspect('equal')
plt.title("u")
plt.scatter(nodes[:,0],nodes[:,1],c=disp[:,0])
plt.colorbar()
plt.subplot(132).set_aspect('equal')
plt.title("v")
plt.scatter(nodes[:,0],nodes[:,1],c=disp[:,1])
plt.colorbar()
plt.subplot(133).set_aspect('equal')
plt.title("shape")
plt.scatter(nodes[:,0],nodes[:,1],c='blue')

extent_x, extent_y = nodes.max(axis=0) - nodes.min(axis=0)
factor = 0.2 * max(extent_x, extent_y)/np.abs(disp).max()
plt.scatter(nodes[:,0]+factor*disp[:,0],nodes[:,1]+factor*disp[:,1],c='red')
plt.show()

# show stress
plt.subplot(131).set_aspect('equal')
plt.title("$\sigma_{xx}$")
plt.scatter(nodes[:,0],nodes[:,1],c=stress[:,0])
plt.colorbar()
plt.subplot(132).set_aspect('equal')
plt.title("$\sigma_{yy}$")
plt.scatter(nodes[:,0],nodes[:,1],c=stress[:,1])
plt.colorbar()
plt.subplot(133).set_aspect('equal')
plt.title("$\sigma_{xy}$")
plt.scatter(nodes[:,0],nodes[:,1],c=stress[:,2])
plt.colorbar()
plt.show()
