import numpy as np

def readInpFileRaw(filename):
    nodes = {}
    nodeSets = {}
    def nodeRead(t):
        i = int(t[0])
        v = tuple(map(float, t[1:]))
        nodes[i] = v
    def nSetRead(t, tag):
        ns = list(map(int, t))
        if tag in nodeSets:
            nodeSets[tag] += ns
        else:
            nodeSets[tag] = ns

    elements = {}
    elementSets = {}
    def elementRead(t, tp, tag):
        i = int(t[0])
        v = tuple(map(int, t[1:]))
        elements[i] = (tp, v)
        # if tag in elementSets:
        #     elementSets[tag] += [i]
        # else:
        #     elementSets[tag] = [i]
    def elSetRead(t, tag):
        els = list(map(int, t))
        if tag in elementSets:
            elementSets[tag] += els
        else:
            elementSets[tag] = els

    actor = [lambda l: None]
    def dispatchSection(head, para):
        if head == 'Heading':
            actor[0] = lambda l: None
            return True
        if head == 'NODE':
            actor[0] = nodeRead
            return True
        if head == 'ELEMENT':
            actor[0] = lambda l, tp=para['type'], tag=para['ELSET']:\
                elementRead(l, tp, tag)
            return True
        if head == 'NSET':
            actor[0] = lambda l, tag=para['NSET']: nSetRead(l, tag)
            return True
        if head == 'ELSET':
            actor[0] = lambda l, tag=para['ELSET']: elSetRead(l, tag)
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
                    actor[0] = lambda t: None
            else: # old part
                actor[0](t)

    return nodes, nodeSets, elements, elementSets


def readInpFile(filename):
    nodes, nodeSets, elements, elementSets = readInpFileRaw(filename)

    # reordering and remove useless
    els = { el : True
            for k in elementSets
            for el in elementSets[k]
            }
    els = list(els)
    els.sort()
    ns = { n : True
            for el in els
            for n in elements[el][1]
            }
    ns = list(ns)
    ns.sort()

    nodes_ = np.array([nodes[o] for o in ns])
    nmap = {o: n for n, o in list(enumerate(ns))}
    for k in elements:
        elements[k] = (elements[k][0], tuple((nmap[o] for o in elements[k][1])))
    for k in nodeSets:
        nodeSets[k] = [nmap[o] for o in nodeSets[k]]

    elements_ = [elements[o] for o in els]
    emap = {o: n for n, o in list(enumerate(els))}
    for k in elementSets:
        elementSets[k] = [emap[o] for o in elementSets[k]]

    return nodes_, nodeSets, elements_, elementSets

if __name__ = '__main__':
    import matplotlib.pyplot as plt

    filename = 'test.inp'
    nodes, nodeSets, elements, elementSets = readInpFile(filename)

    plt.subplot(121).set_aspect('equal')
    for k in list(nodeSets)[::-1]:
        nodeSet = nodes[nodeSets[k]]
        plt.scatter(nodeSet[:,0], nodeSet[:,1])
    plt.legend(list(nodeSets)[::-1], loc='best')

    plt.subplot(122).set_aspect('equal')
    for e in elements:
        nL = nodes[list(e[1])]
        plt.plot(nL[:,0].T, nL[:,1].T)
    plt.show()
