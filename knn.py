from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

SEED = 36


def load_data(real_news_path: str, fake_news_path: str) -> tuple:
    """Loads files in <real_news_path> and <fake_news_path> into a single dataset. 
    Splits the datatest randomly into 70% training, 15% validation, and 15% test examples 
    Returns each example."""

    # read files to lines and store in total_news
    real_news = open(real_news_path, "r").readlines()
    total_news = real_news + open(fake_news_path, "r").readlines()

    # convert total_news to a matrix of token counts
    vectorizer = CountVectorizer()     
    X = vectorizer.fit_transform(total_news)   # need to correct: CountVectorize should have only been fitted on the training data.

    # targets: real or fake news
    y = [1]*len(real_news) + [0] * (len(total_news)-len(real_news))

    # first split dataset to 70% training and 30% other examples,
    # then split other examples to test and validation examples
    X_train, X_other, y_train, y_other = train_test_split(
        X, y, test_size=0.3, random_state=SEED)
    X_test, X_val, y_test, y_val = train_test_split(
        X_other, y_other, test_size=0.5, random_state=SEED)

    return X_train, X_test, X_val, y_train, y_test, y_val


def select_knn_model(data: tuple, metric='minkowski') -> None:
    """uses KNN classifer with different k to classify between real and fake news in <data>.
    Generate plot showing the training and validation accuracy for each k .
    Choose the model with the best validation accuracy and print its k value and test accuracy."""

    k_list = list(range(1, 21))
    X_train, X_test, X_val, y_train, y_test, y_val = data

    t_accuracy = []
    v_accuracy = []
    max_v = 0   # the best validation accuracy
    k_max_v = 1   # k value of the model with the best validation accuracy

    for k in k_list:
        # KNN model and fit it with training data
        knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn_model.fit(X_train, y_train)

        # compute training and validation accuracy
        t_accuracy.append(knn_model.score(X_train, y_train))
        v_acc = knn_model.score(X_val, y_val)
        v_accuracy.append(v_acc)

        # find the best validation accuracy and its k value
        if v_acc > max_v:
            k_max_v = k

    # compute and print the test accuracy of the model with best validation accuracy
    knn_model = KNeighborsClassifier(n_neighbors=k_max_v, metric=metric)
    knn_model.fit(X_train, y_train)
    print('The knn model with k = {} has best validation accuracy, and its accuracy on test data is {}'.format(
        k_max_v, knn_model.score(X_test, y_test)))

    # plot the training and validation accuracy for each k
    plt.plot(k_list, t_accuracy, '-bo')
    plt.plot(k_list, v_accuracy, '-go')
    plt.xticks(np.arange(0, 21, 1))
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('training and validation accuracy for each k')
    plt.legend(['training accuracy', 'validation_accuracy'])
    if metric == 'cosine':
        plt.savefig(
            './1c. training and validation accuracy for each k (metric=cosine).png')
    else:
        plt.savefig('./1b. training and validation accuracy for each k.png')


if __name__ == "__main__":
    real_news_path = "data/clean_real.txt"
    fake_news_path = "data/clean_fake.txt"
    # question 1(a) load data
    data = load_data(real_news_path, fake_news_path)

    # question 1(b)
    select_knn_model(data)

    # question 1(c) pass metric='cosine' to KNeighborsClassifier
    # select_knn_model(data, metric='cosine')
