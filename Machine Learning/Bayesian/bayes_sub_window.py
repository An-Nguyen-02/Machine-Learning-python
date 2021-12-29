import pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
# There are some functions with very small size
# changes from part 1: cifar10_color, cifar_10_naivebayes_learn, mean_std to mean_cov
# new function: run_img_side: a main function implementation as a function so
# I can plot in the main function
# I use the method of flattening all the images to the form of one-hot vector and then calculate it.
# Note: I have checked my function for days but sadly I don't know where it's wrong
# since the code have runtime error when image size 4*4
# Places I put print are places I tried to debug but not yet to find mistakes.
# Therefore, I the probability is only correct for for 1*1,2*2,3*3 images


def unpickle(file):
    with open(file, 'rb') as f:
        dicts = pickle.load(f, encoding="latin1")
    return dicts


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
        img_array = resize_img.getdata()
        img_array = np.reshape(img_array, (1, size*size*3))
        img_array = img_array[0]
        # print(img_array)
        # print(img_array)
        X_img.append(img_array)
    # return converted array of n*n image
    # print(np.array(X_img))
    return np.array(X_img)


# Assign new resize image array to its label by dict
# And return ordered array in term of labels
def dict_label_data(data, labels):
    data_label = {}
    for i in range(np.shape(labels)[0]):
        current_label = labels[i]
        current_img = np.array([data[i]])
        if labels[i] not in data_label.keys():
            data_label[current_label] = current_img
        else:
            data_label[current_label] = np.concatenate((data_label[current_label], current_img), axis=0)
            # print(data_label)
    sorted_array = np.concatenate(
        (data_label[0], data_label[1], data_label[2], data_label[3],
         data_label[4], data_label[5], data_label[6], data_label[7],
         data_label[8], data_label[9]), axis=0)
    # print(np.shape(sorted_array))
    return sorted_array


# return array of mean and covariance matrix for columns of a class
def mean_cov(dataset):
    # Becasue dimension must agree so initiate array with value.
    # put in another array so need to be an 2D array at beginning
    # print(dataset)
    mean_arr = np.array([np.mean(dataset, axis=0)])
    # print(np.shape(mean_arr))
    cov_matrix = np.array([np.cov(dataset, rowvar=False)])
    # print(np.shape(cov_matrix))
    return mean_arr, cov_matrix


# calculate normal distribution probability
def normal_prob(x, mean, covar):
    # print(f"x: {x}")
    # print(f"mean: {mean}")
    # print(f"cov: {covar}")
    probability = multivariate_normal.pdf(x, mean, covar)
    # print(f"probability: {probability}")
    return probability


# Calculate P(x/class) for 1 class
def prob_x_class(x, mean, covar):
    # Function return the normal distribution probability
    n_prob = normal_prob(x, mean, covar)
    return n_prob


# X is sorted array base on class.
def cifar_10_naivebayes_learn(X):
    # each class has same p as 10%
    p = np.repeat([[0.1]], 10, axis=0)
    # Since append in axis = 0, array must be initialized so I take first array as initializer
    class_0_mu, class_0_cov = mean_cov(X[0:5000])
    mu = class_0_mu
    covar = class_0_cov
    class_index = 5000
    for i in range(1, 10):
        sub_arr = X[class_index:class_index+5000]
        class_mu, class_cov = mean_cov(sub_arr)
        mu = np.append(mu, class_mu, axis=0)
        covar = np.append(covar, class_cov, axis=0)
        class_index += 5000
    # print(mu)
    return mu, covar, p


def cifar10_classifier_naivebayes(x, mu, sigma, p):
    const = 10**100
    sigma_x_class = 0
    prob_summary = []
    num_class = 10
    for i in range(num_class):
        curr_mu = mu[i]
        # print(curr_mu)
        curr_sigma = sigma[i]
        # print(np.shape(curr_sigma))
        p_x_class = prob_x_class(x, curr_mu, curr_sigma)*const
        # print(p_x_class)
        sigma_x_class += p_x_class*p[i][0]
    # print(f"hi {sigma_x_class}")
    for i in range(num_class):
        curr_mu = mu[i]
        curr_sigma = sigma[i]
        p_x_class = prob_x_class(x, curr_mu, curr_sigma)*const
        class_prob = p_x_class*p[i][0]/sigma_x_class
        #print(class_prob)
        prob_summary.append(class_prob)
    # print(np.shape(prob_summary))
    highest_chance = max(prob_summary)
    predict_class = prob_summary.index(highest_chance)
    return predict_class, highest_chance


def run_img_size(img_side):
    IMG_SIDE = img_side
    X_train, Y_train = import_train()
    test_num = 10000
    X_test, Y_test = import_test(test_num)
    X_train = cifar10_color(X_train, IMG_SIDE)
    X_test = cifar10_color(X_test, IMG_SIDE)
    #print(X_test)
    #print(X_train)
    X_train = dict_label_data(X_train, Y_train)
    #print(X_train)
    mu, sigma, p = cifar_10_naivebayes_learn(X_train)
    class_predicted = []
    probability_mean = 0
    for i in range(test_num):
        predict_class, predict_prob = cifar10_classifier_naivebayes(X_test[i],
                                                                    mu, sigma,
                                                                    p)
        class_predicted.append(predict_class)
        probability_mean += predict_prob

    probability_mean = probability_mean / test_num
    #print(f"Img size {img_side}*{img_side} has mean class confidence prob: {probability_mean}")
    print(f"Img size {img_side}*{img_side} has class accuracy: {class_acc(class_predicted, Y_test)*100}%")
    return class_acc(class_predicted, Y_test)*100

def main():

    colected_size_acc = []
    img_side = [1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(len(img_side)):
        current_acc = run_img_size(img_side[i])
        colected_size_acc.append(current_acc)

    plt.plot(img_side, colected_size_acc)
    plt.xlabel("Img side")
    plt.ylabel("Accuracy (%)")
    plt.show()






if __name__ == '__main__':
     main()