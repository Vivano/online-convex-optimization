
from utils import *
from gradient_descent import *
# import seaborn as sns
# sns.set_theme()

# Uploading data
train_labels, train_data = prepare_mnist('~/Desktop/m2a/OCO/projet/data/train.csv')
test_labels, test_data = prepare_mnist('~/Desktop/m2a/OCO/projet/data//test.csv')

print(f"Train data shape : {train_data.shape}, Test data shape : {test_data.shape}")
print(f"Train labels shape : {train_labels.shape}, Test labels shape : {test_labels.shape}")

# SVM training parameters
n, d = train_data.shape
n_epochs = 10000
indexes = np.random.choice(n, n_epochs)

# w_ugd = GradientDescent(
#     X=train_data,
#     y=train_labels,
#     projected=False,
#     epochs=n_epochs
# )
# w_pugd = GradientDescent(
#     X=train_data,
#     y=train_labels,
#     z=100,
#     epochs=n_epochs
# )

w_sgd = StochasticGradientDescent(
    X=train_data,
    y=train_labels,
    projected=False,
    epochs=n_epochs
)
w_psgd = StochasticGradientDescent(
    X=train_data,
    y=train_labels,    
    z=50,
    epochs=n_epochs
)

w_smd = StochasticMirrorDescent(
    X=train_data,
    y=train_labels,
    z=50,
    epochs=n_epochs,
)

w_seg = StochasticExponentiatedGradient(
    X=train_data,
    y=train_labels,
    z=100,
    epochs=n_epochs
)

w_adagrad = Adagrad(
    X=train_data,
    y=train_labels,
    z=100,
    epochs=n_epochs
)

w_ons = OnlineNewtonStep(
    X=train_data,
    y=train_labels,
    z=15,
    epochs=n_epochs
)

loss_ugd, loss_pugd = [], []
loss_sgd, loss_psgd = [], []
loss_smd, loss_adagrad, loss_seg = [], [], []
loss_ons = []

for i in range(n_epochs+1):
    # pred_ugd = prediction_svm(w=w_ugd[i], X=test_data)
    # loss_ugd.append(1 - loss01(yhat=pred_ugd, y=test_labels))
    # pred_pugd = prediction_svm(w=w_pugd[i], X=test_data)
    # loss_pugd.append(1 - loss01(yhat=pred_pugd, y=test_labels))

    pred_sgd = prediction_svm(w=w_sgd[i+1], X=test_data)
    loss_sgd.append(1 - loss01(yhat=pred_sgd, y=test_labels))
    pred_psgd = prediction_svm(w=w_psgd[i+1], X=test_data)
    loss_psgd.append(1 - loss01(yhat=pred_psgd, y=test_labels))

    pred_smd = prediction_svm(w=w_smd[i+1], X=test_data)
    loss_smd.append(1 - loss01(yhat=pred_smd, y=test_labels))

    pred_seg = prediction_svm(w=w_seg[i], X=test_data)
    loss_seg.append(1 - loss01(yhat=pred_seg, y=test_labels))

    pred_adagrad = prediction_svm(w=w_adagrad[i], X=test_data)
    loss_adagrad.append(1 - loss01(yhat=pred_adagrad, y=test_labels))

    pred_ons = prediction_svm(w=w_ons[i], X=test_data)
    loss_ons.append(1 - loss01(yhat=pred_ons, y=test_labels))



fig, ax = plt.subplots(figsize=(7,5))
# ax.plot([i+1 for i in range(n_epochs+1)], loss_ugd, label='standard', alpha=0.5)
# ax.plot([i+1 for i in range(n_epochs+1)], loss_pugd, label='projected 100', alpha=0.5)
ax.plot([i+1 for i in range(n_epochs)], loss_sgd, label='sgd', alpha=0.5)
ax.plot([i+1 for i in range(n_epochs)], loss_psgd, label='sgd projected', alpha=0.5)
ax.plot([i+1 for i in range(n_epochs)], loss_smd, label='smd', alpha=0.5)
ax.plot([i+1 for i in range(n_epochs)], loss_seg, label='seg', alpha=0.5)
ax.plot([i+1 for i in range(n_epochs)], loss_adagrad, label='adagrad', alpha=0.5)
ax.plot([i+1 for i in range(n_epochs)], loss_ons, label='ons projected', alpha=0.5)
ax.set_yscale('logit')
# ax.set_xscale('logit') ### TROUVER UN MOYEN DE LE FAIRE MARCHER
ax.set_xscale('log')   # à essayer 
# ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force x axis to integer
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.legend(loc='best')
ax.set_xlabel(r"Epoch $t$")
ax.set_ylabel("Accuracies (decimal)")
ax.set_title("Comparison of all Stochastic Online Optimization algorithms")
plt.show()
