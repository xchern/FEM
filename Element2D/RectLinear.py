import numpy as np

# Gauss quadrature
xs = np.array([-np.sqrt(1/3), np.sqrt(1/3)])
ws = np.array([1, 1])

# Shape function
# N_-1 = (x-1)/2     N'_-1 = x - 1/2
# N_1  = (x+1)/2     N'_1  = x + 1/2
Ns   = np.array([-(xs - 1) / 2, (xs + 1) / 2]) # (node, sample)
NPs = np.array([-1/2 * np.ones_like(xs), 1/2 * np.ones_like(xs)]) # (node, sample)

# 1 |*3 *2
# 0 |*0 *1
#   +-----
#     0  1
order = [(0,0), (1,0), (1,1), (0,1)]

# (node, sample_xi, sample_eta)
N2Ds = np.array([
    np.outer(Ns[i1], Ns[i2])
    for i1, i2 in order
    ])
# (node, sample_xi, sample_eta, {partial xi, partial eta})
NP2Ds = np.array([[[
    (NPs[i1][xii] * Ns[i2][etai], NPs[i2][etai] * Ns[i1][xii])
    for etai, eta in enumerate(xs)]
    for xii, xi in enumerate(xs)]
    for i1, i2 in order]
    )

def CPS4StiffMat(pos, E, nu):
    assert pos.shape == (4,2) #(node, {x,y})
    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2],
        ]) * (E / (1 - nu**2))
    r = np.zeros((8,8))

    # (sample_xi, sample_eta, {partial xi, partial eta}, {x, y})
    Js = np.moveaxis((NP2Ds.T @ pos).T, 0, -1)

    for xii, w1, J_ in zip(range(2), ws, Js):
        for etai, w2, J in zip(range(2), ws, J_):
            NP2D = NP2Ds[:, xii, etai, :] # (node, {partial xi, partial eta})
            NGrad = NP2D @ np.linalg.inv(J.T) # (node, {partial x, partial y})
            B = np.zeros((3, 8))
            B[0,::2] = NGrad[:,0]
            B[1,1::2] = NGrad[:,1]
            B[2,::2] = NGrad[:,1]
            B[2,1::2] = NGrad[:,0]
            r += B.T @ D @ B * (np.linalg.det(J.T) * w1 * w2)
    return r

def CPS4NodalStress(pos, E, nu, disp):
    assert pos.shape == (4,2) #(node, {x,y})
    assert disp.shape == (4,2) #(node, {u,v})
    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2],
        ]) * (E / (1 - nu**2))
    stressS = np.zeros((4, 3))

    # (sample_xi, sample_eta, {partial xi, partial eta}, {x, y})
    Js = np.moveaxis((NP2Ds.T @ pos).T, 0, -1)

    for xii, w1, J_ in zip(range(2), ws, Js):
        for etai, w2, J in zip(range(2), ws, J_):
            NP2D = NP2Ds[:, xii, etai, :] # (node, {partial xi, partial eta})
            NGrad = NP2D @ np.linalg.inv(J.T) # (node, {partial x, partial y})
            B = np.zeros((3, 8))
            B[0,::2] = NGrad[:,0]
            B[1,1::2] = NGrad[:,1]
            B[2,::2] = NGrad[:,1]
            B[2,1::2] = NGrad[:,0]
            stressS[xii*2 + etai] += D @ B @ disp.flatten()

    return np.linalg.pinv(np.reshape(N2Ds,(4,4)).T) @ stressS

def T3D2Force(pos, P):
    assert pos.shape == (2,2)
    # *0  *1
    r = np.zeros(4)
    XP = NPs.T @ pos # (sample point, {x',y'})
    ds = (np.array([[0, -1],[1, 0]]) @ XP.T).T
    for w, N in zip(ws, Ns[0]):
        r[:2] += N * ds[0] * w
    for w, N in zip(ws, Ns[1]):
        r[2:4] += N * ds[1] * w
    r *= P
    return r
