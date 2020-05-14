import numpy as np

entradas = np.array([1, 7, 5])
pesos = np.array([0.8, 0.1, 0])

def soma (e, p):
    return e.dot(p)

def step(soma):
    return 1 if soma >=1 else 0

s = soma(entradas, pesos)
r = step(s)
print(r)