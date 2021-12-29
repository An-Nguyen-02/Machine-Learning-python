import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# Task 2
def class_acc(pred, gt):
    truth_pred = 0
    for i in range(len(pred)):
        if (pred[i] == gt[i]):
            truth_pred += 1
    acc = truth_pred/len(pred)
    return acc

# Import testing data set
datadict = unpickle('cifar-10-batches-py/test_batch')
X = datadict["data"]
Y = datadict["labels"]
Y = np.array(Y)

# Test sample subsetting
num_test = 10000
mask = list(range(num_test))
X_test = X[mask]
Y_test = Y[mask]

# Import the training data set
x_ = []
y_ = []
for i in range(1,6):
    dict_ = unpickle(f"cifar-10-batches-py/data_batch_{i}")
    x_now = dict_["data"]
    y_now = dict_["labels"]
    y_now = np.array(y_now)
    x_.append(x_now)
    y_.append(y_now)

X_train = np.concatenate(x_)
Y_train = np.concatenate(y_)

# Task 3
def cifar10_classifier_random(x):
    return np.random.choice(Y)

# Task 4
def cifar10_classifier_1nn(x,trdata,trlabels):
    # Manhatan distance
    dists = np.sum(np.absolute(x-trdata),axis=1)
    index = np.argmin(dists)
    pred_label = trlabels[index]

    return pred_label

# Following test for task 3
pred3_num = np.array([])
for i in range(X_test.shape[0]):
    pred3_num = np.append(pred3_num, cifar10_classifier_random(X_test.shape[0]))

print(f"Your cifar10_classifier_random accuracy is: {class_acc(pred3_num,Y_test)}")


#Following test for task 4
pred_num = np.array([])
correct_num = np.array([])
for i in range(X_test.shape[0]):
    current_num = cifar10_classifier_1nn(X_test[i], X_train, Y_train)
    pred_num = np.append(pred_num,current_num)
    #print(f"num {Y_test[i]} : predict {current_num}")
    correct_num = np.append(correct_num,Y_test[i])

print(f"Your cifar_classifier_1nn accuracy is: {class_acc(pred_num,correct_num)}")