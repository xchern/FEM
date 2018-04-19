import numpy as np

# Gauss quadrature
xs = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
ws = np.array([5/9, 8/9, 5/9])

point = np.array([(a, b)
    for b in xs
    for a in xs
    ])
weight = np.array([ w1 * w2 for w1 in ws for w2 in ws])

Ns = np.array([
        [ -0.25 * (a + b + 1) * (a - 1) * (b - 1) for a, b in point],
        [ -0.25 * (a - b - 1) * (a + 1) * (b - 1) for a, b in point],
        [ 0.25 * (a + b - 1) * (a + 1) * (b + 1) for a, b in point],
        [ 0.25 * (a - b + 1) * (a - 1) * (b + 1) for a, b in point],
        [ -0.5 * (1 - a ** 2) * (b - 1) for a, b in point],
        [ 0.5 * (1 - b ** 2) * (a + 1) for a, b in point],
        [ 0.5 * (1 - a ** 2) * (b + 1) for a, b in point],
        [ -0.5 * (1 - b ** 2) * (a - 1) for a, b in point],
    ]) # (node, sample)

NP2Ds = np.array([
        [ (-0.25 * (2 * a + b) * (b - 1), -0.25 * (2 * b + a) * (a - 1))
            for a, b in point],
        [ (-0.25 * (2 * a - b) * (b - 1), 0.25 * (2 * b - a) * (a + 1))
            for a, b in point],
        [ (0.25 * (2 * a + b) * (b + 1), 0.25 * (2 * b + a) * (a + 1))
            for a, b in point],
        [ (0.25 * (2 * a - b) * (b + 1), -0.25 * (2 * b - a) * (a - 1))
            for a, b in point],
        [ (a*(b-1), -0.5*(1-a**2)) for a, b in point],
        [ (0.5*(1-b**2), -(a+1)*b) for a, b in point],
        [ (-(b+1)*a, 0.5*(1-a**2)) for a, b in point],
        [ (-0.5*(1-b**2), (a-1)*b) for a, b in point],
    ]) # (node, sample, {p_a,p_b})

## check
# import matplotlib.pyplot as plt
# for N, NP2D in zip(Ns, NP2Ds):
#     plt.quiver(point[:,0], point[:,1], NP2D[:,0], NP2D[:,1])
#     plt.imshow(np.reshape(N, (3,3)),
#             extent=(-xs[2] * 1.5,xs[2] * 1.5,-xs[2] * 1.5,xs[2] * 1.5),
#             origin='lower')
#     plt.show()

def CPS8StiffMat(pos, E, nu):
    assert pos.shape == (8,2) #(node, {x,y})
    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2],
        ]) * (E / (1 - nu**2))
    r = np.zeros((16,16))

    # (sample, {partial a, partial b}, {x, y})
    Js = np.moveaxis((NP2Ds.T @ pos).T, 0, -1)

    for (a, b), w, J, si in zip(point, weight, Js, range(9)):
            NP2D = NP2Ds[:, si, :] # (node, {partial a, partial b})
            NGrad = NP2D @ np.linalg.inv(J.T) # (node, {partial x, partial y})
            B = np.zeros((3, 16))
            B[0,::2] = NGrad[:,0]
            B[1,1::2] = NGrad[:,1]
            B[2,::2] = NGrad[:,1]
            B[2,1::2] = NGrad[:,0]
            r += B.T @ D @ B * (np.linalg.det(J.T) * w)
    return r

def CPS8NodalStress(pos, E, nu, disp):
    assert pos.shape == (8,2) #(node, {x,y})
    assert disp.shape == (8,2) #(node, {u,v})
    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2],
        ]) * (E / (1 - nu**2))

    stressS = np.zeros((9,3))

    # (sample, {partial a, partial b}, {x, y})
    Js = np.moveaxis((NP2Ds.T @ pos).T, 0, -1)

    for (a, b), w, J, si in zip(point, weight, Js, range(9)):
            NP2D = NP2Ds[:, si, :] # (node, {partial a, partial b})
            NGrad = NP2D @ np.linalg.inv(J.T) # (node, {partial x, partial y})
            B = np.zeros((3, 16))
            B[0,::2] = NGrad[:,0]
            B[1,1::2] = NGrad[:,1]
            B[2,::2] = NGrad[:,1]
            B[2,1::2] = NGrad[:,0]
            stressS[si] += D @ B @ disp.flatten()

    return np.linalg.pinv(Ns.T) @ stressS
