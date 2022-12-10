
from utils import *
from sklearn.svm import SVC
import gradient_descent

# Uploading data
train_labels, train_data = prepare_mnist('../data/train.csv')
test_labels, test_data = prepare_mnist('../data/test.csv')

print(f"Train data shape : {train_data.shape}, Test data shape : {test_data.shape}")
print(f"Train labels shape : {train_labels.shape}, Test labels shape : {test_labels.shape}")

# SVM training parameters
lbd = 1/3
n_epochs = 50
lr = np.array([1 / (lbd*(i+1)) for i in range(n_epochs)])

clf = mySVM(
    lambda_=lbd,
    n_iter=n_epochs,
    learning_rate=lr, 
    optim_method='ugd'
)

perf = clf.fit(train_data, train_labels)
# pred_train = clf.predict(train_data)
# pred_test = clf.predict(test_data)
fig, ax = plt.subplots(1, 2)
fig, ax = plot_perf(fig, ax, perf)
plt.show()
#print(perf)