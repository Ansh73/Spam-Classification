import os
import re
from pathlib import Path
import math
import datetime

if __name__ == '__main__':
    ham_cp_dict = {'ham': {}}
    spam_cp_dict = {'spam': {}}
    vocabulary_dict = {}
    ham_vocabulary_dict = {}
    spam_vocabulary_dict = {}
    ham_denominator = spam_denominator = 0
    index_of_words = 1
    directory_of_files = Path("./hw2_train/train")
    directory_of_test_files = Path("./hw2_test/test")
    # directory_of_files = Path("./hw2_train/Dummy_Train")
    # directory_of_test_files = Path("./hw2_test/Dummy_Test")

    for dirpath, dirs, files in os.walk(directory_of_files):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            with open(fname, encoding='Latin-1') as f:
                for line in f:
                    for word in re.findall(r'\w+', line):
                        if word not in vocabulary_dict:
                            vocabulary_dict[word] = index_of_words
                            index_of_words += 1
                        if 'ham' in fname:
                            if word not in ham_vocabulary_dict:
                                ham_vocabulary_dict[word] = 1
                            else:
                                ham_vocabulary_dict[word] += 1
                        else:
                            if word not in spam_vocabulary_dict:
                                spam_vocabulary_dict[word] = 1
                            else:
                                spam_vocabulary_dict[word] += 1
# get the smoothed count of ham words
for ham_words in vocabulary_dict:
    value = ham_vocabulary_dict.get(ham_words, 0)
    ham_denominator += value + 1
# get the smoothed count of spam words
for spam_words in vocabulary_dict:
    value = spam_vocabulary_dict.get(spam_words, 0)
    spam_denominator += value + 1

# get the probability of words belonging to the classes
for word in vocabulary_dict:
    ham_numerator = ham_vocabulary_dict.get(word, 0)
    ham_numerator = ham_numerator + 1
    ham_prob = ham_numerator / ham_denominator
    ham_cp_dict['ham'][word] = ham_prob

    spam_numerator = spam_vocabulary_dict.get(word, 0)
    spam_numerator = spam_numerator + 1
    spam_prob = spam_numerator / spam_denominator
    spam_cp_dict['spam'][word] = spam_prob

# Test the learned probabilities
total_files = 0
total_ham_files = total_spam_files = 0

for dirpath, dirs, files in os.walk(directory_of_test_files):
    if 'ham' in str(dirpath):
        total_ham_files = len(files)
    else:
        total_spam_files = len(files)

total_files = total_ham_files + total_spam_files

prior_probability_ham = total_ham_files / total_files
prior_probability_spam = total_spam_files / total_files
correctly_classified = 0
log_prior_ham = 0
log_prior_spam = 0
for dirpath, dirs, files in os.walk(directory_of_test_files):
    for filename in files:
        fname = os.path.join(dirpath, filename)
        with open(fname, encoding='Latin-1') as f:
            if prior_probability_ham != 0:
                log_prior_ham = math.log(prior_probability_ham, 2)
            if prior_probability_spam != 0:
                log_prior_spam = math.log(prior_probability_spam, 2)
            likelihood_ham = 0
            likelihood_spam = 0
            posterior_ham = 0
            posterior_spam = 0
            for line in f:
                for word in re.findall(r'\w+', line):
                    likelihood_ham += math.log(ham_cp_dict['ham'].get(word, 1))
                    likelihood_spam += math.log(spam_cp_dict['spam'].get(word, 1))

            posterior_ham = log_prior_ham + likelihood_ham
            posterior_spam = log_prior_spam + likelihood_spam
            if 'ham' in fname:
                if posterior_ham > posterior_spam:
                    correctly_classified += 1
            elif 'spam' in fname:
                if posterior_spam > posterior_ham:
                    correctly_classified += 1

# get accuracy spam
total_accuracy = correctly_classified / total_files * 100
print("The accuracy is " + str(total_accuracy))


# sigmoid function
def sigmoid(weights, row):
    len_row = len(row)
    z = weights[0]
    for i in range(1, len_row - 1):
        if weights[i] != 0 and row[i] != 0:
            z += weights[i] * row[i]
    sigmoid_denominator = 1 + math.exp(-z)
    return 1 / sigmoid_denominator
    return 0


# gradient ascent for updating weights
def gradient_ascent(list_of_temp_weights, input_data, col_number):
    gradient_ascent_value = 0
    total_rows = len(input_data)
    total_col = len(input_data[0])
    for k in range(0, total_rows):
        print("Gradient ascent for weight  " + str(col_number) + " For row " + str(k))
        gradient_ascent_value += input_data[k][col_number] * (input_data[k][total_col - 1] -
                                                              sigmoid(list_of_temp_weights, input_data[k]))
    return gradient_ascent_value


def logistic_regression(list_of_weights_to_learn, input_values, learning_rate, lambda_param):
    for i in range(0, 1):
        for j in range(len(list_of_weights_to_learn)):
            print("Lerning weight  " + str(j) + " For itr  " + str(i))
            list_of_weights_to_learn[j] = list_of_weights[j] + \
                                          learning_rate * gradient_ascent(list_of_weights_to_learn, input_values, j) \
                                          - learning_rate * lambda_param * list_of_weights[j]


# Train the logistic function with training data


total_train_files = 0
total_train_ham_files = 0
total_train_spam_files = 0
for dirpath, dirs, files in os.walk(directory_of_files):
    if 'ham' in str(dirpath):
        total_train_ham_files = len(files)
    else:
        total_train_spam_files = len(files)

total_train_files = total_train_ham_files + total_train_spam_files

list_of_keys = [(key, 0) for key in vocabulary_dict]
# my_ordered_dict_of_words = OrderedDict(list_of_keys)

temp_dict_of_words_and_freq = {}
list_of_weights = [0 for i in range(len(list_of_keys) + 1)]
input_data = [0] * total_train_files
test_data = [0] * 1
weighted_sum = 0
output_column = len(list_of_keys) + 1
for i in range(len(input_data)):
    input_data[i] = [0] * (len(list_of_keys) + 2)
    input_data[i][0] = 1
for t in range(len(test_data)):
    test_data[t] = [0] * (len(list_of_keys) + 1)
    test_data[t][0] = 1

row_index = 0
# Loop through files one by one and store frequency of words in the matrix created above
for dirpath, dirs, files in os.walk(directory_of_files):
    for filename in files:
        fname = os.path.join(dirpath, filename)
        with open(fname, encoding='Latin-1') as f:
            for line in f:
                for word in re.findall(r'\w+', line):
                    if temp_dict_of_words_and_freq.get(word, 0) == 0:
                        temp_dict_of_words_and_freq[word] = 1
                    else:
                        temp_dict_of_words_and_freq[word] += 1
                    temp_index = vocabulary_dict.get(word)
                    input_data[row_index][temp_index] = temp_dict_of_words_and_freq[word]
            if 'ham' in fname:
                input_data[row_index][output_column] = 0
            else:
                input_data[row_index][output_column] = 1
            row_index += 1
            temp_dict_of_words_and_freq.clear()
# print("List of weights before regularization ", list_of_weights)
logistic_regression(list_of_weights, input_data, 0.001, 0.005)
# print("List of weights after regularization ", list_of_weights)

logistic_correcly_classified = 0

for dirpath, dirs, files in os.walk(directory_of_test_files):
    for filename in files:
        fname = os.path.join(dirpath, filename)
        with open(fname, encoding='Latin-1') as f:
            for line in f:
                for word in re.findall(r'\w+', line):
                    if word not in temp_dict_of_words_and_freq:
                        temp_dict_of_words_and_freq[word] = 1
                    else:
                        temp_dict_of_words_and_freq[word] += 1
                    temp_index = vocabulary_dict.get(word)
                    test_data[0][temp_index] = temp_dict_of_words_and_freq[word]
            # probability calculation
            p = list_of_weights[0]
            for j in range(1, len(test_data[0])):
                p += list_of_weights[j] * test_data[0][j]

            if p < 0.5:
                if 'ham' in fname:
                    logistic_correcly_classified += 1
            else:
                if 'spam' in fname:
                    logistic_correcly_classified += 1
            row_index += 1
            temp_dict_of_words_and_freq.clear()

logistic_accuracy = logistic_correcly_classified / total_files
print("Logistic accuracy is " + str(logistic_accuracy))
