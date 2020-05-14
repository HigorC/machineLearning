import numpy as np

# AND
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([0, 0, 0, 1])
# OR
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([0, 1, 1, 1])
pesos = np.array([0.0, 0.0])
taxaAprendizagem = 0.1

def step(soma):
    return 1 if soma >=1 else 0

def calculaSaida(registro):
    s = registro.dot(pesos)
    return step(s)

def treinar():
    erroTotal = 1
    while (erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.array(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro

            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print('Peso atualizado: ' + str(pesos[j]))

        print('Total de erros: ' + str(erroTotal))

treinar();