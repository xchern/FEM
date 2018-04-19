import numpy as np

point = [(1/6, 1/6), (1/6, 2/3), (2/3, 1/6)]
weight = [1/6, 1/6, 1/6]
Ns = np.array([
        [ 2 * (a + b - 0.5) * (a + b - 1) for a, b in point],
        [ 2 * a * (a - 0.5) for a, b in point],
        [ 2 * b * (b - 0.5) for a, b in point],
        [ -4 * a * (a + b - 1) for a, b in point],
        [ 4 * a * b for a, b in point],
        [ -4 * b * (a + b - 1) for a, b in point],
    ]) # (node, sample)
NP2Ds = np.array([
        [ (4 * (a + b) - 3, 4 * (a + b) - 3)
            for a, b in point],
        [ (4 * a - 1, 0)
            for a, b in point],
        [ (0, 4 * b - 1)
            for a, b in point],
        [ (-4 * (2 * a + b - 1), -4 * a)
            for a, b in point],
        [ (4 * b, 4 * a)
            for a, b in point],
        [ (-4 * b, -4 * (2 * b + a - 1))
            for a, b in point],
    ]) # (node, sample, {p_a,p_b})


def CPS6StiffMat(pos, E, nu):
    assert pos.shape == (6,2) #(node, {x,y})
    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2],
        ]) * (E / (1 - nu**2))
    r = np.zeros((12,12))

    # (sample, {partial a, partial b}, {x, y})
    Js = np.moveaxis((NP2Ds.T @ pos).T, 0, -1)

    for (a,b), w, J, si in zip(point, weight, Js, range(3)):
            NP2D = NP2Ds[:, si, :] # (node, {partial a, partial b})
            NGrad = NP2D @ np.linalg.inv(J.T) # (node, {partial x, partial y})
            B = np.zeros((3, 12))
            B[0,::2] = NGrad[:,0]
            B[1,1::2] = NGrad[:,1]
            B[2,::2] = NGrad[:,1]
            B[2,1::2] = NGrad[:,0]
            r += B.T @ D @ B * (np.linalg.det(J.T) * w)

    return r

NsLower = np.array([
        [ -(a + b - 1) for a, b in point],
        [ a for a, b in point],
        [ b for a, b in point],
    ]) # (node, sample)

def CPS6NodalStress(pos, E, nu, disp):
    assert pos.shape == (6,2) #(node, {x,y})
    assert disp.shape == (6,2) #(node, {u,v})

    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2],
        ]) * (E / (1 - nu**2))

    stressS = np.zeros((3,3))

    # (sample, {partial a, partial b}, {x, y})
    Js = np.moveaxis((NP2Ds.T @ pos).T, 0, -1)

    for (a,b), w, J, si in zip(point, weight, Js, range(3)):
            NP2D = NP2Ds[:, si, :] # (node, {partial a, partial b})
            NGrad = NP2D @ np.linalg.inv(J.T) # (node, {partial x, partial y})
            B = np.zeros((3, 12))
            B[0,::2] = NGrad[:,0]
            B[1,1::2] = NGrad[:,1]
            B[2,::2] = NGrad[:,1]
            B[2,1::2] = NGrad[:,0]
            stressS[si] = D @ B @ disp.flatten()

    stress = np.linalg.inv(NsLower.T) @ stressS
    return stress
