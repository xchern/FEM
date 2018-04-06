import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from readInp import readInpFile
from Element2D import CPS9StiffMat, T3D3Force

filename = 'test.inp'

nodes, nodeSets, elements, elementSets = readInpFile(filename)
nodes = nodes[:,:2]

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
            assert tp == 'CPS4' and len(ns) == 9
            subMat = CPS9StiffMat(nodes[ns], para['E'], para['nu'])
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
            assert tp == 'T3D3' and len(ns) == 3
            subRhs = T3D3Force(nodes[ns], para['P'])
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
            assert tp == 'T3D3' and len(ns) == 3
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

# show load
plt.subplot(121).set_aspect('equal')
plt.title("F_u")
plt.scatter(nodes[:,0],nodes[:,1],c=rhs[::2])
plt.colorbar()
plt.subplot(122).set_aspect('equal')
plt.title("F_v")
plt.scatter(nodes[:,0],nodes[:,1],c=rhs[1::2])
plt.colorbar()
plt.show()

print("solving linear system")
import scipy.sparse.linalg

# TODO which is better?
x = sparse.linalg.spsolve(mat, rhs)
# x, e = sparse.linalg.cgs(mat, rhs)
# print('error', e)

x.shape = (nN, 2)

# show result
plt.subplot(121).set_aspect('equal')
plt.title("u")
plt.scatter(nodes[:,0],nodes[:,1],c=x[:,0])
plt.colorbar()
plt.subplot(122).set_aspect('equal')
plt.title("v")
plt.scatter(nodes[:,0],nodes[:,1],c=x[:,1])
plt.colorbar()
plt.show()
