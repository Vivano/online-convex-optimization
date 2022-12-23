
from utils import *
from sklearn.svm import SVC
import gradient_descent
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
# import seaborn as sns
# sns.set_theme()

# Uploading data
train_labels, train_data = prepare_mnist('~/Desktop/m2a/OCO/projet/data/train.csv')
test_labels, test_data = prepare_mnist('~/Desktop/m2a/OCO/projet/data//test.csv')

print(f"Train data shape : {train_data.shape}, Test data shape : {test_data.shape}")
print(f"Train labels shape : {train_labels.shape}, Test labels shape : {test_labels.shape}")

# SVM training parameters
lbd = 1/3
gamma = 1/8
n_epochs = 50
lr = np.array([1 / (lbd*(i+1)) for i in range(n_epochs)])

# clf = mySVM(
#     lambda_=lbd,
#     n_iter=n_epochs,
#     learning_rate=lr, 
#     optim_method='ugd'
# )
# perf = clf.fit(train_data, train_labels)
# # pred_train = clf.predict(train_data)
# # pred_test = clf.predict(test_data)
# fig, ax = plt.subplots(1, 2)
# fig, ax = plot_perf(fig, ax, perf)
# plt.show()
# #print(perf)

w_ugd = gradient_descent.unconstrained_gd(
    epochs=n_epochs,
    eta=lr,
    lambda_=lbd,
    X=train_data,
    y=train_labels
)

w_pugd = gradient_descent.projected_unconstrained_gd(
    epochs=n_epochs,
    eta=lr,
    z=100,
    lambda_=lbd,
    X=train_data,
    y=train_labels
)

w_sgd = gradient_descent.SGD(
    epochs=n_epochs,
    lambda_=lbd,
    X=train_data,
    Y=train_labels
)

w_psgd = gradient_descent.projected_SGD(
    epochs=n_epochs,
    z=100,
    lambda_=lbd,
    X=train_data,
    Y=train_labels
)

w_smd = gradient_descent.SMD(
    epochs=n_epochs,
    z=100,
    lambda_=lbd,
    X=train_data,
    Y=train_labels
)

w_seg = gradient_descent.SEG(
    epochs=n_epochs,
    z=100,
    lambda_=lbd,
    X=train_data,
    Y=train_labels
)

w_adagrad = gradient_descent.Adagrad(
    epochs=n_epochs,
    z=100,
    lambda_=lbd,
    X=train_data,
    Y=train_labels
)

w_ons = gradient_descent.ONS(
    epochs=n_epochs,
    z=100,
    lambda_=lbd,
    gamma = gamma,
    X=train_data,
    Y=train_labels
)

loss_ugd, loss_pugd = [], []
loss_sgd, loss_psgd = [], []
loss_smd, loss_adagrad, loss_seg = [], [], []
loss_ons = []
# loss_svm = []
for i in range(n_epochs+1):
    pred_ugd = prediction_svm(w=w_ugd[i], X=test_data)
    loss_ugd.append(loss01(yhat=pred_ugd, y=test_labels))
    pred_pugd = prediction_svm(w=w_pugd[i], X=test_data)
    loss_pugd.append(loss01(yhat=pred_pugd, y=test_labels))

    pred_sgd = prediction_svm(w=w_sgd[i], X=test_data)
    loss_sgd.append(loss01(yhat=pred_sgd, y=test_labels))
    pred_psgd = prediction_svm(w=w_psgd[i], X=test_data)
    loss_psgd.append(loss01(yhat=pred_psgd, y=test_labels))

    pred_smd = prediction_svm(w=w_smd[i], X=test_data)
    loss_smd.append(loss01(yhat=pred_smd, y=test_labels))
    pred_seg = prediction_svm(w=w_seg[i], X=test_data)
    loss_seg.append(loss01(yhat=pred_seg, y=test_labels))
    pred_adagrad = prediction_svm(w=w_adagrad[i], X=test_data)
    loss_adagrad.append(loss01(yhat=pred_adagrad, y=test_labels))

    pred_ons = prediction_svm(w=w_ons[i], X=test_data)
    loss_ons.append(loss01(yhat=pred_ons, y=test_labels))

fig, ax = plt.subplots(figsize=(7,5))
ax.plot([i for i in range(n_epochs+1)], loss_ugd, label='standard', alpha=0.5)
ax.plot([i for i in range(n_epochs+1)], loss_pugd, label='projected', alpha=0.5)
ax.plot([i for i in range(n_epochs+1)], loss_sgd, label='sgd', alpha=0.5)
ax.plot([i for i in range(n_epochs+1)], loss_psgd, label='sgd projected', alpha=0.5)
ax.plot([i for i in range(n_epochs+1)], loss_smd, label='smd', alpha=0.5)
ax.plot([i for i in range(n_epochs+1)], loss_seg, label='seg', alpha=0.5)
ax.plot([i for i in range(n_epochs+1)], loss_adagrad, label='adagrad', alpha=0.5)
ax.plot([i for i in range(n_epochs+1)], loss_ons, label='ons projected', alpha=0.5)
ax.set_yscale('logit')
#ax.set_xscale('logit') ### TROUVER UN MOYEN DE LE FAIRE MARCHER
#ax.set_xscale('log')   # à essayer 


ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force x axis to integer
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.legend(loc='best')
plt.show()
