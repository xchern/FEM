import numpy as np

def CPS3StiffMat(pos, E, nu):
    assert pos.shape == (3,2) #(node, {u,v})
    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2],
        ]) * (E / (1 - nu**2))

    r = np.zeros((6,6))

    pos01 = pos[0] - pos[1]
    pos12 = pos[1] - pos[2]
    pos20 = pos[2] - pos[0]

    grad0 = pos01 - pos12.dot(pos01)/pos12.dot(pos12) * pos12
    grad0 /= grad0.dot(grad0)
    grad1 = pos12 - pos20.dot(pos12)/pos20.dot(pos20) * pos20
    grad1 /= grad1.dot(grad1)
    grad2 = pos20 - pos01.dot(pos20)/pos01.dot(pos01) * pos01
    grad2 /= grad2.dot(grad2)

    NGrad = np.array([grad0, grad1, grad2]) # (node, {partial x, partial y})

    B = np.zeros((3, 6))
    B[0,::2] = NGrad[:,0]
    B[1,1::2] = NGrad[:,1]
    B[2,::2] = NGrad[:,1]
    B[2,1::2] = NGrad[:,0]

    r += B.T @ D @ B * np.abs(np.linalg.det(np.array((pos01, pos12)))) / 2

    return r
