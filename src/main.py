
from utils import *
from sklearn.svm import SVC
import gradient_descent

# Uploading data
train_labels, train_data = prepare_mnist('../data/train.csv')
test_labels, test_data = prepare_mnist('../data/test.csv')

print(f"Train data shape : {train_data.shape}, Test data shape : {test_data.shape}")
print(f"Train labels shape : {train_labels.shape}, Test labels shape : {test_labels.shape}")

# SVM training parameters
lbd = 1.
C = 1.
n_epochs = 100
lr = np.array([1e-1 for i in range(n_epochs)])

clf = mySVM(
    lambda_=lbd,
    C=C,
    n_iter=n_epochs,
    gradient_descent=gradient_descent.unconstrained_gd,
    learning_rate=lr
)
clf_projected = mySVM(
    lambda_=lbd,
    C=C,
    n_iter=n_epochs,
    gradient_descent=gradient_descent.projected_unconstrained_gd,
    learning_rate=lr
)

clf.fit(train_data, train_labels)
pred_train = clf.predict(train_data)
pred_test = clf.predict(test_data)
print(f"Train/Acc={accuracy(pred_train, train_labels)}, Test/Acc={accuracy(pred_test, test_labels)}")

clf_projected.fit(train_data, train_labels)
pred_train = clf_projected.predict(train_data)
pred_test = clf_projected.predict(test_data)
print(f"Train/Acc={accuracy(pred_train, train_labels)}, Test/Acc={accuracy(pred_test, test_labels)}")