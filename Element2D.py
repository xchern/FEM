import numpy as np

# Gauss quadrature
xs = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
ws = np.array([5/9, 8/9, 5/9])

# Shape function
# N_-1 = (x-1)x/2     N'_-1 = x - 1/2
# N_0  = 1-x^2        N'_0  = -2x
# N_1  = (x+1)x/2     N'_1  = x + 1/2
Ns   = np.array([(xs - 1) * xs / 2, 1- xs ** 2, (xs + 1) * xs / 2]) # (node, sample)
NPs = np.array((xs - 1/2, -2 * xs, xs + 1/2)) # (node, sample)

# 2 |*3  *6  *2
# 1 |*7  *8  *5
# 0 |*0  *4  *1
#   +----------
#     0   1   2
order = [(0,0), (2,0), (2,2), (0,2),
        (1,0), (2,1), (1,2), (0,1),
        (1,1)]

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

def CPS9StiffMat(pos, E, nu):
    assert pos.shape == (9,2) #(node, {u,v})
    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2],
        ]) * (E / (1 - nu**2))
    r = np.zeros((18,18))

    # (sample_xi, sample_eta, {partial xi, partial eta}, {x, y})
    Js = np.moveaxis((NP2Ds.T @ pos).T, 0, -1)

    for xii, w1, J_ in zip(range(3), ws, Js):
        for etai, w2, J in zip(range(3), ws, J_):
            NP2D = NP2Ds[:, xii, etai, :] # (node, {partial xi, partial eta})
            NGrad = NP2D @ np.linalg.inv(J.T) # (node, {partial x, partial y})
            B = np.zeros((3, 18))
            B[0,::2] = NGrad[:,0]
            B[1,1::2] = NGrad[:,1]
            B[2,::2] = NGrad[:,1]
            B[2,1::2] = NGrad[:,0]
            r += B.T @ D @ B * (np.linalg.det(J.T) * w1 * w2)
    return r

def T3D3Force(pos, P):
    assert pos.shape == (3,2)
    # very strange order
    # *0  *1  *2
    r = np.zeros(6)
    XP = NPs.T @ pos # (sample point, {x',y'})
    ds = (np.array([[0, 1],[-1, 0]]) @ XP.T).T
    for w, N in zip(ws, Ns[0]):
        r[:2] += N * ds[0] * w
    for w, N in zip(ws, Ns[1]):
        r[2:4] += N * ds[1] * w
    for w, N in zip(ws, Ns[2]):
        r[4:] += N * ds[2] * w
    r *= P
    return r
