import numpy as np


def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)


entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([[0], [1], [1], [0]])

# pesos0 = np.array([[-0.424, -0.740, -0.961],
#                    [0.358, -0.577, -0.469]])

pesos1 = np.array([[-0.017], [-0.893], [0.148]])


# Gera pesos aleatórios para  iniciar a rede
# O -1 é para que todos os valores fiquem negativos
# Junto com o *2, alguns dos valores ficaram positivos
pesos0 = 2 * np.random.random((2,3)) - 1
pesos1 = 2 * np.random.random((3,1)) - 1

epocas = 10000
taxaAprendizagem = 0.7
momento = 1


for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    # print(somaSinapse0)
    # print(camadaOculta)
    # print(camadaSaida)
    # print(camadaSaida - saidas)


    # Python faz várias contas de uma vez
    # Aqui está subtraindo cada posição no vetor pelo outro
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))

    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida

    pesosTransposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesosTransposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)