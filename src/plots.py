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

    # xval_z(
    #     name='ONS', 
    #     algo=OnlineNewtonStep, 
    #     X=train_data, 
    #     y=train_labels, 
    #     z_list=[10, 50, 100, 1000],
    #     Epochs=10**4
    # )

    # xval_lbd(
    #     name='projected GD', 
    #     algo=GradientDescent, 
    #     projected=True,
    #     X=train_data, 
    #     y=train_labels, 
    #     lambda_list=[1., 1/3, 0.1, 0.01],
    #     Epochs=1000
    # )

    # comparison_sgd(
    #     name="projected GD", 
    #     algo=GradientDescent, 
    #     algo_sgd=StochasticGradientDescent, 
    #     Xtrain=train_data, Ytrain=train_labels,
    #     Xtest=test_data, Ytest=test_labels,
    #     z_algo=50, z_sgd=50, 
    #     Epochs=10**4)

    # comparison_sgd(
    #     name="SMD", 
    #     algo=StochasticMirrorDescent, 
    #     algo_sgd=StochasticGradientDescent, 
    #     Xtrain=train_data, Ytrain=train_labels,
    #     Xtest=test_data, Ytest=test_labels,
    #     z_algo=50, z_sgd=50, 
    #     Epochs=10**4)

    # comparison_sgd(
    #     name="SEG", 
    #     algo=StochasticExponentiatedGradient, 
    #     algo_sgd=StochasticGradientDescent, 
    #     Xtrain=train_data, Ytrain=train_labels,
    #     Xtest=test_data, Ytest=test_labels,
    #     z_algo=100, z_sgd=50, 
    #     Epochs=10**4)

    comparison_sgd(
        name="Adagrad", 
        algo=Adagrad, 
        algo_sgd=StochasticGradientDescent, 
        Xtrain=train_data, Ytrain=train_labels,
        Xtest=test_data, Ytest=test_labels,
        z_algo=100, z_sgd=50, 
        Epochs=10**4)