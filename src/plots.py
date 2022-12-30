from utils import *
from gradient_descent import *

train_labels, train_data = prepare_mnist('~/Desktop/m2a/OCO/projet/data/train.csv')
test_labels, test_data = prepare_mnist('~/Desktop/m2a/OCO/projet/data//test.csv')

if __name__=="__main__":

    # xval_z(
    #     name='GD', 
    #     algo=GradientDescent, 
    #     X=train_data, 
    #     y=train_labels, 
    #     z_list=[10, 50, 100, 1000],
    #     Epochs=10**2
    # )

    # xval_z(
    #     name='SGD', 
    #     algo=StochasticGradientDescent, 
    #     X=train_data, 
    #     y=train_labels,
    #     z_list=[10, 50, 100, 1000],
    #     Epochs=10**4
    # )

    # xval_z(
    #     name='SMD', 
    #     algo=StochasticMirrorDescent, 
    #     X=train_data, 
    #     y=train_labels, 
    #     z_list=[10, 50, 100, 1000],
    #     Epochs=10**4
    # )

    # xval_z(
    #     name='SEGpm', 
    #     algo=StochasticExponentiatedGradient, 
    #     X=train_data, 
    #     y=train_labels, 
    #     z_list=[10, 50, 100, 1000],
    #     Epochs=10**4
    # )

    # xval_z(
    #     name='Adagrad', 
    #     algo=Adagrad, 
    #     X=train_data, 
    #     y=train_labels, 
    #     z_list=[10, 50, 100, 1000],
    #     Epochs=10**4
    # )

    xval_z(
        name='ONS', 
        algo=OnlineNewtonStep, 
        X=train_data, 
        y=train_labels, 
        z_list=[10, 50, 100, 1000],
        Epochs=10**4
    )
