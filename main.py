import numpy as np
from scipy import sparse

from readInp import readInpFile
from Element import CPS9StiffMat, T3D3Force

filename = 'test.inp'

nodes, _, elements, elementSets = readInpFile(filename)

nN = len(nodes)

print('constructing linear system...')
# construct linear system
val = []
row = []
col = []
rhs = np.zeros(nN * 3)

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
            subMat = CPS9StiffMat(nodes[ns], para['G'], para['nu'])
            indices = [i for n in ns for i in (3 * n, 3 * n + 1, 3 * n + 2)]
            c, r = np.meshgrid(indices, indices)
            col += c.flatten().tolist()
            row += r.flatten().tolist()
            val += subMat.flatten().tolist()
mat = sparse.coo_matrix((val, (row, col)), shape=(nN * 3, nN * 3))

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
            indices = [i for n in ns for i in (3 * n, 3 * n + 1, 3 * n + 2)]
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
                    indices.append(ni * 3)
            if 'v' in para:
                assert para['v'] == 0
                for ni in ns:
                    indices.append(ni * 3 + 1)
            if 'w' in para:
                assert para['w'] == 0
                for ni in ns:
                    indices.append(ni * 3 + 2)
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

print("solving linear system")
import scipy.sparse.linalg

# TODO which is better?
# x = sparse.linalg.spsolve(mat, rhs)
x, e = sparse.linalg.cgs(mat, rhs)
print('error', e)

x.shape = (nN, 3)
