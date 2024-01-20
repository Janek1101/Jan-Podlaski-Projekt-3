import numpy as np
import matplotlib.pyplot as plt

def Euler(eps1, eps2, gam1, gam2, h1, h2, n1_init, n2_init):
    N1 = [n1_init]
    N2 = [n2_init]

    h = 0.001
    time = np.arange(0, 20, h)

    for i in range(1, time.shape[0]):
        n1 = N1[-1] + h * ((eps1 - gam1 * (h1 * N1[-1] + h2 * N2[-1])) * N1[-1])
        n2 = N2[-1] + h * ((eps2 - gam2 * (h1 * N1[-1] + h2 * N2[-1])) * N2[-1])

        N1.append(n1)
        N2.append(n2)
            
    return [N1, N2]

if __name__ == '__main__':
    K = 100000  
    r = .4   

    h = 0.01
    time = np.arange(75, 120, h) 

    gompertz = [10]
    verhulst = [10]

    for i in range(1, time.shape[0]):
        gomp = gompertz[-1] + h * r * gompertz[-1] * np.log(K/gompertz[-1])
        verh = verhulst[-1] + h * r * verhulst[-1] * (1 - verhulst[-1]/K) 
        gompertz.append(gomp)
        verhulst.append(verh)

    plt.plot(time, gompertz, label='Równanie Gompertza')
    plt.plot(time, verhulst, label='Równanie Verhulsta')
    plt.ylabel('Populacja')
    plt.xlabel('Czas')
    plt.title('Porównanie równania Gompertza i równania Verhulsta')
    plt.legend()
    plt.show() 

    h = 0.001
    time = np.arange(0, 20, h)

    n1, n2 = Euler(1.25, .5, .5, .2, .1, .2, 3, 4)

    plt.plot(time, n1, label='N1')
    plt.plot(time, n2, label='N2')
    plt.ylabel('Populacja')
    plt.xlabel('Czas')
    plt.title('Porównanie wcześniej zdefiniowanych wartości 1')
    plt.legend()
    plt.show()

    n1, n2 = Euler(5, 5, 4, 8, 1, 4, 3, 4)

    plt.plot(time, n1, label='N1')
    plt.plot(time, n2, label='N2')
    plt.ylabel('Populacja')
    plt.xlabel('Czas')
    plt.title('Porównanie wcześniej zdefiniowanych wartości 2')
    plt.legend()
    plt.show()

    eps1 = .8
    gam1 = 1
    h1 = .3

    eps2 = .4
    gam2 = .5
    h2 = .4

    c1, c2 = Euler(eps1, eps2, gam1, gam2, h1, h2, 4, 8)
    d1, d2 = Euler(eps1, eps2, gam1, gam2, h1, h2, 8, 8)
    e1, e2 = Euler(eps1, eps2, gam1, gam2, h1, h2, 12, 8)

    x = np.linspace(0, 13, 26)
    y = np.linspace(0, 13, 26)

    X, Y = np.meshgrid(x, y)
    dX = np.zeros(X.shape)
    dY = np.zeros(Y.shape)

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            dX[i, j] = (eps1 - gam1 * (h1 * X[i, j] + h2 * Y[i, j])) * X[i, j]
            dY[i, j] = (eps2 - gam2 * (h1 * X[i, j] + h2 * Y[i, j])) * Y[i, j]

    plt.quiver(X, Y, dX, dY)
    plt.plot(c1, c2, label='N1 = 4, N2 = 8')
    plt.plot(d1, d2, label='N1 = 8, N2 = 8')
    plt.plot(e1, e2, label='N1 = 12, N2 = 8')
    plt.legend()
    plt.xlabel('Populacja n1')
    plt.ylabel('Populacja n2')
    plt.title('Wykres fazowy z trzema krzywymi')
    plt.show()
