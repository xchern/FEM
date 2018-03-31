import numpy as np

#Gauss quadrature
# g_x = [0]
# g_w = [2]
# g_x = [-np.sqrt(1/3), np.sqrt(1/3)]
# g_w = [1, 1]
g_x = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
g_w = [5/9, 8/9, 5/9]

def CPS9StiffMat(pos, G, nu):
    assert pos.shape == (9,3)
    # TODO
    return np.eye(27) - 1e-3

def T3D3Force(pos, P):
    assert pos.shape == (3,3)
    # TODO
    return np.ones(9)
