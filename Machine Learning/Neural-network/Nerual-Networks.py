import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
import numpy as np
# The following are accuracies I got so far on all test data.
# 1nn: 27.31%
# Bayesian best: 41.83%
# Neural network: 50.04%

def unpickle(file):
    with open(file, 'rb') as f:
        dicts = pickle.load(f, encoding="latin1")
    return dicts

# Import the training data set
def import_train():
    dict_1 = unpickle(f"cifar-10-batches-py/data_batch_{1}")
    X_train = dict_1["data"]
    Y_train = np.array([dict_1["labels"]])
    Y_train = np.transpose(Y_train)
    for i in range(2, 6):
        dict_ = unpickle(f"cifar-10-batches-py/data_batch_{i}")
        x_now = dict_["data"]
        y_now = np.array([dict_["labels"]])
        y_now = np.transpose(y_now)
        X_train = np.append(X_train,x_now,axis=0)
        Y_train = np.append(Y_train,y_now,axis=0)

    return X_train/255, Y_train

# Import testing data
def import_test(sub_test):
    # Import testing data set
    datadict = unpickle('cifar-10-batches-py/test_batch')
    X = datadict["data"]
    Y = np.array([datadict["labels"]])
    Y = np.transpose(Y)

    # Test sample subsetting
    mask = list(range(sub_test))
    X_test = X[mask]
    Y_test = Y[mask]
    return X_test/255, Y_test

def convert_one_hot(y):
    num_sample = np.shape(y)[0]
    y_one_hot = np.zeros([num_sample,10])
    for i in range(num_sample):
        current_class = y[i][0]
        y_one_hot[i][current_class] = 1
    return y_one_hot

def neural_predict(x_train,y_train,x_test,y_test):
    model = Sequential()
    # sigmoid, softmax,exponential

    model.add(Dense(40, input_dim=3072, activation='elu'))
    model.add(Dense(30,activation='sigmoid'))
    model.add(Dense(20,activation='elu'))
    model.add(Dense(10,activation='sigmoid'))
    opt = tf.keras.optimizers.SGD(learning_rate=0.005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    y_1 = convert_one_hot(y_train)
    model.fit(x_train, y_1, epochs=100, verbose=1)
    y_pred = np.empty(y_test.shape)
    y_pred_2 = np.squeeze(model.predict(x_test))
    for pred_ind in range(y_pred_2.shape[0]):
        current_arr = y_pred_2[pred_ind]
        best_class = np.argmax(current_arr)
        y_pred[pred_ind] = best_class
    tot_correct = len(np.where(y_test-y_pred == 0)[0])
    accuracy = tot_correct / len(y_test) * 100
    print(
        f'Classication accuracy: {accuracy}%')
    return accuracy

def main():

    x_train, y_train = import_train()
    train_acc = neural_predict(x_train,y_train,x_train,y_train)
    x_test, y_test = import_test(10000)
    #train_acc = neural_predict(x_train,y_train,x_test,y_test)
    # predict train

if __name__ == '__main__':
     main()