# Preimenujte datoteko v solution.py, da bodo testi delovali

import math
import matplotlib.pyplot as plt
import numpy
import sys
from decimal import Decimal
from operator import itemgetter
from scipy.optimize import fmin_l_bfgs_b



def load(name):
    """ 
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke) 
    in vektor razredov.
    """
    data = numpy.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(numpy.int)
    return X, y


def h(x, theta):
    """ 
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """

    return 1 / (1 + math.e ** (-x.dot(theta)))


def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    # ... dopolnite (naloga 1, naloga 2)
    m = len(y)
    gladkost = lambda_ * sum([el ** 2 for el in theta]) / (2 * m)
    prilagajanje = [
        yi * numpy.log((h(xi, theta))) + (1 - yi) * numpy.log((1 - h(xi, theta))) for
        xi, yi in zip(X, y)]
    return -1 / (m) * (sum(prilagajanje)) + gladkost


def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne numpyev vektor v velikosti vektorja theta.
    """
    res = numpy.zeros(len(theta))
    m = len(y)
    for i, el in enumerate(theta):
        res[i] = 1 / m * sum([(h(xi, theta) - yi) * xi[i] for xi, yi in zip(X, y)]) + lambda_ * el / m

    return res


class LogRegClassifier(object):
    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = numpy.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetno razreda 1
        return [1 - p1, p1]


class LogRegLearner(object):
    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = numpy.hstack((numpy.ones((len(X), 1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(cost,
                              x0=numpy.zeros(X.shape[1]),
                              args=(X, y, self.lambda_),
                              fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X, y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, k=5):
    order = list(range(len(y)))
    numpy.random.shuffle(order)
    X = my_shuffle(X, order)
    y = my_shuffle(y, order)
    """
    c = list(zip(X, y))
    numpy.random.shuffle(c)
    X, y = zip(*c)
    X=list(X)
    y=list(y)

    """
    koef = int(len(y) / k)
    Xlist = []
    ylist = []
    tmp_pos = 0
    ostanek = len(y) - koef * k
    for i in range(k):
        start = tmp_pos
        end = tmp_pos + koef
        if ostanek > 0:
            end += 1
            ostanek -= 1
        Xlist.append((X[start:end]))
        ylist.append(y[start:end])
        tmp_pos = end

    res = []

    for i in range(k):
        trainy = []
        trainx = []

        for j in range(k):
            if j != i:
                trainy.extend(ylist[j])
                trainx.extend(Xlist[j])

        testX = Xlist[i]

        classifier = learner(numpy.asarray(trainx), numpy.asarray(trainy))
        for el in testX:
            napoved = classifier(el)
            res.append(napoved)
    reverse_order = list(range(len(y)))
    for i in range(len(y)):
        reverse_order[order[i]] = i

    res = my_shuffle(res, reverse_order)
    return numpy.asarray(res)


def my_shuffle(seznam, order):
    seznam = [seznam[i] for i in order]
    return seznam


def CA(real, predictions, threshold=0.5):
    # Classification accuracy
    # ... dopolnite (naloga 3)
    s = 0
    for i in range(len(real)):
        if real[i] == 0 and predictions[i][0] >= threshold:
            s += 1
        elif real[i] == 1 and predictions[i][0] < threshold:
            s += 1
    return s / len(real)


def compute_trapez(v1, v2, x1, x2):
    S = (Decimal(v2) + Decimal(v1)) / Decimal(2)
    return S * (Decimal(x1) - Decimal(x2))




def set_classes(probabilities, t):
    res = numpy.zeros(len(probabilities))
    for i, el in enumerate(probabilities):
        if el < t:
            res[i] = 1
        else:
            res[i] = 0
    return res


def conf_matrix(real, predictions, t):
    predictions = set_classes(predictions, t)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(real)):
        if real[i] == 1:
            TP += 1 - predictions[i]
            FN += predictions[i]
        else:
            TN += predictions[i]
            FP += 1 - predictions[i]

    return TP, TN, FP, FN


def AUC(real, predictions):
    #določim thresholde na normalizirane na intervalu 0-1
    all_values = []
    for el in predictions:
        all_values.extend(list(el))

    all_values = list(set(all_values))
    all_values.sort()

    new_pred = (predictions - min(all_values)) / (max(all_values) - min(all_values))
    new_pred = [d[1] for d in new_pred] #potrebujemo samo prvo vrednost

    #uredimo poradtke po vrsti
    new_order = numpy.argsort(new_pred, kind="mergesort")[::-1]
    new_pred = my_shuffle(new_pred, new_order)
    real = my_shuffle(real, new_order)

    # izračunamo senzitivity in specifity za vsak threshold
    results = []
    for t in new_pred:
        (TP, TN, FP, FN) = conf_matrix(real, new_pred, t)
        sp = TN / max((TN + FP), 0.001)
        se = TP / max((TP + FN), 0.001)
        results.append((sp, se))

    # izračun površine - trapez
    S = 0
    for el in range(len(results) - 1):
        x1 = results[el][0]
        x2 = results[el + 1][0]
        y1 = results[el][1]
        y2 = results[el + 1][1]

        S += compute_trapez(y1, y2, x1, x2)

    return float(S)


if __name__ == "__main__":
    # Primer uporabe

    X, y = load('reg.data')
    la = 0.00001
    for j in range(5):
        learner = LogRegLearner(lambda_=la)
        res = test_cv(learner, X, y)
        print("Lambda: ", la, " Tocnost:", CA(y, res))
        la *= 100
