import numpy as np

# Données XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Initialisation
np.random.seed(0)
W1 = np.random.randn(2,4)
b1 = np.zeros((1,4))
W2 = np.random.randn(4,1)
b2 = np.zeros((1,1))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)

lr = 0.1

# Entraînement
for epoch in range(2000):

    # Propagation avant
    a1 = sigmoid(np.dot(X,W1)+b1)
    y_pred = sigmoid(np.dot(a1,W2)+b2)

    # Calcul de la perte
    loss = np.mean((y - y_pred)**2)

    # Rétropropagation
    d2 = (y_pred - y)*sigmoid_deriv(y_pred)
    d1 = np.dot(d2,W2.T)*sigmoid_deriv(a1)

    # Mise à jour des poids
    W2 -= lr*np.dot(a1.T,d2)
    b2 -= lr*np.sum(d2,axis=0,keepdims=True)
    W1 -= lr*np.dot(X.T,d1)
    b1 -= lr*np.sum(d1,axis=0,keepdims=True)

    if epoch % 500 == 0:
        print("Loss:", loss)

print("\nRésultat final :")
print(y_pred.round())

''' Explication du processus d’apprentissage

Lors de la propagation avant, les données d’entrée passent à travers les couches du réseau, où elles sont transformées par des poids et des fonctions d’activation pour produire une sortie.
L’erreur est ensuite calculée en comparant la prédiction du modèle avec la vraie valeur à l’aide d’une fonction de perte (ici l’erreur quadratique moyenne).
Pendant la rétropropagation, le réseau calcule les gradients de l’erreur par rapport aux poids en remontant du résultat vers l’entrée.
Ces gradients sont utilisés pour ajuster les poids afin de réduire l’erreur.
Au fil des itérations, les poids s’améliorent et le modèle devient plus précis.'''