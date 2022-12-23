
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


loss_ugd, loss_pugd = [], []
# loss_svm = []
for i in range(n_epochs+1):
    pred_ugd = prediction_svm(w=w_ugd[i], X=test_data)
    loss_ugd.append(loss01(yhat=pred_ugd, y=test_labels))
    pred_pugd = prediction_svm(w=w_pugd[i], X=test_data)
    loss_pugd.append(loss01(yhat=pred_pugd, y=test_labels))

fig, ax = plt.subplots(figsize=(7,5))
ax.plot([i for i in range(n_epochs+1)], loss_ugd, label='standard', alpha=0.5)
ax.plot([i for i in range(n_epochs+1)], loss_pugd, label='projected', alpha=0.5)
ax.set_yscale('logit')
# ax.set_xscale('logit') ### TROUVER UN MOYEN DE LE FAIRE MARCHER
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force x axis to integer
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.legend(loc='best')
plt.show()