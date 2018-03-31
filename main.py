import numpy as np
import matplotlib.pyplot as plt

filename = 'test-rl.inp'

nodes = []
def nodeRead(t):
    i = int(t[0])
    v = tuple(map(float, t[1:]))
    nodes.append(v)

elementTypes = (
        'CPS3',
        'CPS6',

        'CPS4',
        'CPS8',
        )
elements = {}
def elementRead(t, tp):
    i = int(t[0])
    v = tuple(map(int, t[1:]))
    if tp in elements:
        elements[tp].append(v)
    else:
        elements[tp] = [v]

nodeSetTags = (
        'FixU',
        'FixV',
        'Pressure',
        )
#nodeSetTags = ['FixU', 'FixV']
nodeSets = {}
def nsetRead(t, tag):
    ns = list(map(int, t))
    if tag in nodeSets:
        nodeSets[tag] += ns
    else:
        nodeSets[tag] = ns



actor = lambda l: None
def dispatchSection(head, para):
    global actor
    if head == 'NODE':
        actor = nodeRead
        return True
    if head == 'ELEMENT' and para['type'] in elementTypes:
        actor = lambda l, tp=para['type']: elementRead(l, tp)
        return True
    if head == 'NSET' and para['NSET'] in nodeSetTags:
        actor = lambda l, tag=para['NSET']: nsetRead(l, tag)
        return True
    return False

with open(filename) as fo:
    while True:
        l = fo.readline()
        if not l:
            break
        l = l[:-1] # remove '\n'
        if l[:2] == '**': continue # comment
        # get tokens
        t = l.replace(' ', '')
        t = t.split(',')
        if t[-1] == '': t.pop()

        if t[0][0] == '*': # new part
            head = t[0][1:]
            para = {k:v for p in t[1:]
                    for k, v in (p.split('='),)
                    }
            if not dispatchSection(head, para):
                print("ignoring unknown part", l)
                actor = lambda t: None
        else: # old part
            actor(t)

# convert to array
nodes = np.array(nodes)
for k in elements:
    elements[k] = np.array(elements[k]) - 1 # -1 for python index from 0 rather than 1
for k in nodeSets:
    nodeSets[k] = np.array(nodeSets[k]) - 1 # -1 for python index from 0 rather than 1


plt.subplot(121).set_aspect('equal')
plt.scatter(nodes[:,0], nodes[:,1])
for k in nodeSets:
    nodeSet = nodes[nodeSets[k]]
    plt.scatter(nodeSet[:,0], nodeSet[:,1])
plt.legend(['all'] + list(nodeSets), loc='best')
for k in nodeSets:
    nodeSet = nodes[nodeSets[k]]
    plt.plot(nodeSet[:,0], nodeSet[:,1])

plt.subplot(122).set_aspect('equal')
for k in elements:
    elementL = nodes[elements[k]]
    plt.plot(elementL[:,:,0].T, elementL[:,:,1].T)
plt.show()
