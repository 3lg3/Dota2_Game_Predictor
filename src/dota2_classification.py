import numpy as np
import csv
from sklearn.svm import SVC  # "Support vector classifier"
import matplotlib.pyplot as plt


def measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == -1:
            TN += 1
        if y_hat[i] == -1 and y_actual[i] != y_hat[i]:
            FN += 1
    return TP, FP, TN, FN


def bootstrapping(B, X, y, C_p, gamma_p, K):
    # B: number of times to do bootstrapping
    # C_p: slack variable of SVM
    # gamma_p: gamma of SVM
    # K: kernel of SVM

    n = len(X)
    d = len(X[0])
    z = np.zeros((B, 1))
    bs_err = np.zeros(B)
    for i in range(0, B):
        u = [0] * n
        S = []  # sampling
        for j in range(0, n):
            k = np.random.randint(n)
            u[j] = k
            if not (k in S):
                S.append(k)
        Se = []
        for k in range(0, n):
            Se.append(k)
        T = list(set(Se) - set(S))  # testing
        alg = SVC(C=C_p, kernel=K, gamma=gamma_p)
        alg.fit(X[S], y[S])
        bs_err[i] = np.mean(y[T] != alg.predict(X[T]))
    err = np.mean(bs_err)
    return err


reader = csv.reader(open("dota2.csv", "rt"), delimiter=",")
fbuffer = list(reader)
x = np.array(fbuffer).astype("int")
x = x[0:2000]
y = x[:, 0]
y = y.ravel()
x = np.delete(x, 0, 1)
x = np.delete(x, 0, 1)
x = np.delete(x, 0, 1)
x = np.delete(x, 0, 1)

# we are not interested in game mode, cluster, etc for now and instead just focusing on the heros picked each game, so the first column indicating
# the result of the game plus the following three columns which are about general game info are removed.


positive_y = []
negative_y = []
for i in range(0, 2000):
    if y[i] == 1:
        positive_y.append(i)
    else:
        negative_y.append(i)

positive_len = len(positive_y)
negative_len = len(negative_y)

print ("positive:")
print (positive_len)
print ("negative:")
print (negative_len)
print ("\n")

print ("SVM with rbf kernel:")

samples_fold1 = positive_y[0:int(positive_len / 2)] + negative_y[0:int(negative_len / 2)]
samples_fold2 = positive_y[int(positive_len / 2):] + negative_y[int(negative_len / 2):]

C_list = [0.1, 1, 10]
B = 30
gamma_list = [0.01, 0.05, 0.1]
# first have a fixed c of 1 , do hyperparameter tuning for gamma

sensitivity_rbf = []
specificity_rbf = []

best_gamma = 0
best_error = 1.1
for gamma in gamma_list:
    err = bootstrapping(B, x[samples_fold1], y[samples_fold1], 1, gamma, "rbf")
    print ("gamma =", gamma, ",error = ", err)
    if err < best_error:
        best_error = err
        best_gamma = gamma
best_C = 0
best_error = 1.1
# best gamma obtained, do hyperparameter tuning for C based on this gama
for slack in C_list:
    err = bootstrapping(B, x[samples_fold1], y[samples_fold1], slack, best_gamma, "rbf")
    print ("C =", slack, ",error = ", err)
    if err < best_error:
        best_error = err
        best_C = slack
print ("Results of hyperparameter tuning: (trained by x[samples_fold1], y[samples_fold1]")
print ("C: ", best_C)
print ("gamma: ", best_gamma)
print ("\n")

# cross validation
y_pred = np.zeros(len(x), int)
alg = SVC(C=best_C, kernel="rbf", gamma=best_gamma)
alg.fit(x[samples_fold1], y[samples_fold1])
y_pred[samples_fold2] = alg.predict(x[samples_fold2])
# y_pred is the list of predicted label of x in samples_fold1

for gamma in gamma_list:
    alg = SVC(C=best_C, kernel="rbf", gamma=gamma)
    alg.fit(x[samples_fold1], y[samples_fold1])
    y_predict_temp = alg.predict(x[samples_fold2])
    tp, fp, tn, fn = measure(y[samples_fold2], y_predict_temp)
    # print "y_predict_temp"
    # print y_predict_temp
    sensitivity_rbf.append(1.0 * tp / (tp + fn))
    specificity_rbf.append(1.0 * tn / (tn + fp))

best_gamma = 0
best_error = 1.1
for gamma in gamma_list:
    err = bootstrapping(B, x[samples_fold2], y[samples_fold2], 1, gamma, "rbf")
    print ("gamma =", gamma, ",error = ", err)
    if err < best_error:
        best_error = err
        best_gamma = gamma
best_C = 0
best_error = 1.1
# best gamma obtained, do hyperparameter tuning for C based on this gama
for slack in C_list:
    err = bootstrapping(B, x[samples_fold2], y[samples_fold2], slack, best_gamma, "rbf")
    print ("C =", slack, ",error = ", err)
    if err < best_error:
        best_error = err
        best_C = slack
print ("Results of hyperparameter tuning: (trained by x[samples_fold2], y[samples_fold2]")
print ("C: ", best_C)
print ("gamma: ", best_gamma)

for gamma in gamma_list:
    alg = SVC(C=best_C, kernel="rbf", gamma=gamma)
    alg.fit(x[samples_fold2], y[samples_fold2])
    y_predict_temp = alg.predict(x[samples_fold1])
    tp, fp, tn, fn = measure(y[samples_fold1], y_predict_temp)
    sensitivity_rbf.append(1.0 * tp / (tp + fn))
    specificity_rbf.append(1.0 * tn / (tn + fp))

# cross validation
alg = SVC(C=best_C, kernel="rbf", gamma=best_gamma)
alg.fit(x[samples_fold2], y[samples_fold2])
y_pred[samples_fold1] = alg.predict(x[samples_fold1])
error = np.mean(y != y_pred)
rbf_error = error

print ("\n")
print ("error rate of SVM with a rbf kernel: ", error)
print ("\n\n\n")
print ("SVM with linear kernel:")

sensitivity_linear = []
specificity_linear = []

best_C = 0
best_error = 1.1
# best gamma obtained, do hyperparameter tuning for C based on this gama
for slack in C_list:
    err = bootstrapping(B, x[samples_fold1], y[samples_fold1], slack, "auto", "linear")
    print ("C =", slack, ",error = ", err)
    if err < best_error:
        best_error = err
        best_C = slack
print ("Results of hyperparameter tuning: (trained by x[samples_fold1], y[samples_fold1]")
print ("C: ", best_C)
print ("\n")

for slack in C_list:
    alg = SVC(C=slack, kernel="linear", gamma="auto")
    alg.fit(x[samples_fold1], y[samples_fold1])
    y_predict_temp = alg.predict(x[samples_fold2])
    tp, fp, tn, fn = measure(y[samples_fold2], y_predict_temp)
    sensitivity_linear.append(1.0 * tp / (tp + fn))
    specificity_linear.append(1.0 * tn / (tn + fp))

# cross validation
y_pred = np.zeros(len(x), int)
alg = SVC(C=best_C, kernel="linear", gamma="auto")
alg.fit(x[samples_fold1], y[samples_fold1])
y_pred[samples_fold2] = alg.predict(x[samples_fold2])

best_C = 0
best_error = 1.1
# best gamma obtained, do hyperparameter tuning for C based on this gama
for slack in C_list:
    err = bootstrapping(B, x[samples_fold2], y[samples_fold2], slack, "auto", "linear")
    print ("C =", slack, ",error = ", err)
    if err < best_error:
        best_error = err
        best_C = slack
print ("Results of hyperparameter tuning: (trained by x[samples_fold2], y[samples_fold2]")
print ("C: ", best_C)

for slack in C_list:
    alg = SVC(C=slack, kernel="linear", gamma="auto")
    alg.fit(x[samples_fold2], y[samples_fold2])
    y_predict_temp = alg.predict(x[samples_fold1])
    tp, fp, tn, fn = measure(y[samples_fold1], y_predict_temp)
    sensitivity_linear.append(1.0 * tp / (tp + fn))
    specificity_linear.append(1.0 * tn / (tn + fp))

# cross validation
alg = SVC(C=best_C, kernel="linear", gamma="auto")
alg.fit(x[samples_fold2], y[samples_fold2])
y_pred[samples_fold1] = alg.predict(x[samples_fold1])

error = np.mean(y != y_pred)

print ("\n")
print ("error rate of SVM with a linear kernel: ", error)
print ("error rate of SVM with rbf kernel: ", rbf_error)

print ("sensitivity of rbf kernel on fold1:")
print (sensitivity_rbf[0:3])
print ("specificity of rbf kernel on fold 1:")
print (specificity_rbf[0:3])
print ("sensitivity of rbf kernel on fold2:")
print (sensitivity_rbf[3:6])
print ("specificity of rbf kernel on fold 2:")
print (specificity_rbf[3:6])

print ("sensitivity of linear kernel on fold1:")
print (sensitivity_linear[0:3])
print ("specificity of linear kernel on fold 1:")
print (specificity_linear[0:3])
print ("sensitivity of linear kernel on fold2:")
print (sensitivity_linear[3:6])
print ("specificity of linear kernel on fold 2:")
print (specificity_linear[3:6])

print ("program finished. loading graph...")

freq = np.zeros(len(x[0]), int)
for i in range(0, 2000):
    for j in range(0, len(x[0])):
        if x[i][j] != 0:
            freq[j] = freq[j] + 1
plt.plot(freq)
plt.show()
