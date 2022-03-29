import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd

dat =pd.read_csv('iris.csv')
data = dat.sample(150)
print(data)
print(data.describe())
print(data.shape)

X1 = data.iloc[:, :2].values
X2 = data.iloc[:, 2:5].values

plt.scatter(X1[:,0],X1[:,1],c='black')
plt.xlabel('Comprimento Septico')
plt.ylabel('Largura Septica')
plt.savefig("Banco1.png")
plt.show()

plt.scatter(X2[:,0],X2[:,1],c='black')
plt.xlabel('Comprimento da Petala')
plt.ylabel('Largura da Petala')
plt.savefig("Banco2.png")
plt.show()

def kmeans(dataset, c1, c2, namec1, namec2, save):
    X = dataset.iloc[:, [c1,c2]].values
    m = X.shape[0]
    n = X.shape[1]
    K= 3
    centroids = np.array([]).reshape(n,0)
    for i in range(K):
        rand = rd.randint(0,m-1)
        centroids = np.c_[centroids,X[rand]]

    interacoes = 100
    resultado = {}

    for n in range(interacoes):
        Euclid = np.array([]).reshape(m, 0)
        for r in range(K):
            total = np.sum((((X - centroids[:,r])** 2 ) ** 1/2), axis=1)
            Euclid = np.c_[Euclid, total]

        C = np.argmin(Euclid, axis=1) +1
        Y = {}
        for k in range(K):
            Y[k + 1] = np.array([]).reshape(2, 0)
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]], X[i]]
        for k in range(K):
            Y[k + 1] = Y[k + 1].T
        for k in range(K):
            centroids[:, k] = np.mean(Y[k + 1], axis=0)

        resultado = Y


    color = ['blue', 'green', 'purple']

    labels = ['Cluster1', 'Cluster2', 'Cluster3']

    for k in range(K):
           plt.scatter(resultado[k + 1][:, 0], resultado[k + 1][:, 1], c=color[k], label=labels[k])
    plt.scatter(centroids[0, :], centroids[1, :], s=100, c='red', label='Centroides')
    plt.xlabel(namec1)
    plt.ylabel(namec2)
    plt.legend()
    plt.savefig(save)
    plt.show()

    return  resultado

result1 = kmeans(data, 0,1, 'Comprimento Septico', 'Largura Septica', "imagem1.png" )

result2 = kmeans(data, 2, 3, 'Comprimento da Petala','Largura da Petala', "imagem2.png")

