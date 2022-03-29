import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd

data=pd.read_csv('tripadvisor_review.csv')
print(data.describe())



def kmeans(dataset, c1, c2, namec1, namec2, save):
    X = dataset.iloc[:, [c1,c2]].values
    m = X.shape[0]
    n = X.shape[1]
    K= 3

    X = dataset.iloc[:, [1,2]].values

    m = X.shape[0]
    n = X.shape[1]
    K=4

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

    color=['blue','green','yellow','purple']
    labels=['Cluster1','Cluster2','Cluster3','Cluster4']
    for k in range(K):
        plt.scatter(resultado[k + 1][:, 0], resultado[k + 1][:, 1], c=color[k], label=labels[k])
    plt.scatter(centroids[0, :], centroids[1, :], s=100, c='red', label='Centroides')
    plt.xlabel(namec1)
    plt.ylabel(namec2)
    plt.legend()
    plt.savefig(save)
    plt.show()

    return resultado

result1 = kmeans(data, 1, 2, 'Feedback médio do usuário em galerias de arte',
                 'Feedback médio do usuário em danceterias', 'imagem3.png')

result2 = kmeans(data, 3, 4, 'Feedback médio do usuário em bares de suco',
                 'Feedback médio do usuário em restaurantes', 'imagem4.png')

result3 = kmeans(data, 5, 7, 'Feedback médio do usuário em museus',
                 'Feedback médio do usuário em parques / piqueniques', 'imagem5.png')

result4 = kmeans(data, 9, 10, ' Feedback médio do usuário nos cinemas',
                 'Feedback médio do usuário em instituições religiosas', 'imagem6.png')