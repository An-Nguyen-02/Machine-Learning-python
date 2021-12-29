import pickle
from PIL import Image
import numpy as np

import math
# The result obtained is 19.41%


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# Import the training data set
def import_train():
    dict_1 = unpickle(f"cifar-10-batches-py/data_batch_{1}")
    X_train = dict_1["data"]
    Y_train = dict_1["labels"]
    for i in range(2, 6):
        dict_ = unpickle(f"cifar-10-batches-py/data_batch_{i}")
        x_now = dict_["data"]
        y_now = dict_["labels"]
        y_now = np.array(y_now)
        X_train = np.append(X_train,x_now,axis=0)
        Y_train = np.append(Y_train,y_now)

    return X_train, Y_train


# Import testing data
def import_test(sub_test):
    # Import testing data set
    datadict = unpickle('cifar-10-batches-py/test_batch')
    X = datadict["data"]
    Y = datadict["labels"]
    Y = np.array(Y)

    # Test sample subsetting
    mask = list(range(sub_test))
    X_test = X[mask]
    Y_test = Y[mask]
    return X_test, Y_test


# Return class accuracy
def class_acc(pred, gt):
    truth_pred = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            truth_pred += 1
    acc = truth_pred/len(pred)
    return acc


# Convert input dataset to image, resize them,
# convert back to RGB array and return them.
def cifar10_color(dataset, size=1):

    num_sample = np.shape(dataset)[0]
    dataset = dataset.reshape(num_sample, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    X_img = []
    for i in range(np.shape(dataset)[0]):
        new_img = Image.fromarray(dataset[i])
        resize_img = new_img.resize([size, size])
        img_array = np.array(resize_img.getdata())
        X_img.append(img_array)
    # return converted array of n*n image
    return np.array(X_img)


# Assign new resize image array to its label by dict
# And return ordered array in term of labels
def dict_label_data(data, labels):
    data_label = {}
    for i in range(np.shape(labels)[0]):
        current_label = labels[i]
        if labels[i] not in data_label.keys():
            data_label[current_label] = data[i]
        else:
            data_label[current_label] = np.concatenate(
                (data_label[current_label], data[i]), axis=0)
    sorted_array = np.concatenate(
        (data_label[0], data_label[1], data_label[2], data_label[3],
         data_label[4], data_label[5], data_label[6], data_label[7],
         data_label[8], data_label[9]), axis=0)
    return sorted_array


# return array of mean and standard deviataion for columns of dataset
def mean_std(dataset):
    # Becasue dimension must when I use append in cifar_10_naivebayes_learn
    # So i need to put another array outside below mean and std
    mean_arr = np.array([np.mean(dataset, axis=0)])
    #print(mean_arr)
    std_arr = np.array([np.std(dataset, axis=0)])
    return mean_arr, std_arr


# calculate normal distribution probability
def normal_prob(x, mean, std):
    exponent = math.exp(-((x-mean)**2 / (2 * std**2)))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent


# Calculate P(x/class) for 1 class
def prob_x_class(x, mean, std):
    # Function return the normal distribution probability
    n_prob = 1
    for color in range(np.shape(x)[1]):
        # print(f"{x[color]},{mean[color]},{std[color]}")
        # Because x is 2D array so the index become x[0][color] instead of x[color]
        n_curr_color = normal_prob(x[0][color], mean[color], std[color])
        n_prob *= n_curr_color
    return n_prob


# X is sorted array base on class.
def cifar_10_naivebayes_learn(X):
    # each class has same p as 10%
    p = np.repeat([[0.1]], 10, axis=0)
    # Since append in axis = 0, array must be initialized by the first class value
    class_0 = X[0:5000, :]
    mu, sigma = mean_std(class_0)
    #print(mu)
    #print(sigma)
    class_index = 5000
    for i in range(1, 10):
        sub_arr = X[class_index:class_index+5000, :]
        class_mu, class_sigma = mean_std(sub_arr)
        mu = np.append(mu, class_mu, axis=0)
        sigma = np.append(sigma, class_sigma, axis=0)
        class_index += 5000
    return mu, sigma, p


def cifar10_classifier_naivebayes(x, mu, sigma, p):
    sigma_x_class = 0
    prob_summary = []
    num_class = 10
    for i in range(num_class):
        curr_mu = mu[i]
        # print(curr_mu)
        curr_sigma = sigma[i]
        p_x_class = prob_x_class(x, curr_mu, curr_sigma)
        sigma_x_class += p_x_class*p[i][0]

    for i in range(num_class):
        curr_mu = mu[i]
        curr_sigma = sigma[i]
        p_x_class = prob_x_class(x, curr_mu, curr_sigma)
        class_prob = p_x_class*p[i][0]/sigma_x_class
        prob_summary.append(class_prob)

    highest_chance = max(prob_summary)
    predict_class =  prob_summary.index(highest_chance)
    return predict_class, highest_chance


def main():
    X_train, Y_train = import_train()
    #print(f"a {X_train}")
    test_num = 10000
    X_test, Y_test = import_test(test_num)
    #print(f"b {X_test}")
    X_train = cifar10_color(X_train)
    X_test = cifar10_color(X_test)
    # print(X_test[0])
    X_train = dict_label_data(X_train, Y_train)
    #print(X_train[0:5000])
    mu, sigma, p = cifar_10_naivebayes_learn(X_train)
    class_predicted = []
    probability_mean = 0
    for i in range(test_num):
        predict_class, predict_prob = cifar10_classifier_naivebayes(X_test[i]
                                                                , mu, sigma, p)
        class_predicted.append(predict_class)
        probability_mean += predict_prob

    probability_mean = probability_mean/test_num
    print(f"Average confidence of result is {probability_mean*100}%")
    print(f"The class accuracy is {class_acc(class_predicted,Y_test)*100}%")


if __name__ == '__main__':
    main()