from solution import *
from draw import *
import pylab
import csv

X, y = load('reg.data')
grid = 50


def napacne_slike(real, predictions, images):
    res = []
    for i in range(len(real)):
        if real[i] == 1 and predictions[i][0] >= 0.5:
            res.append(images[i]+" "+str(predictions[i]))
        elif real[i] == 0 and predictions[i][0] < 0.5:
            res.append(images[i]+" "+str(predictions[i]))
    return res



# NALOGA 2

learner = LogRegLearner(lambda_=0.0)
classifier = learner(X, y)
draw_decision(X, y, classifier, 0, 1, grid, "Lambda 0", 1)

learner = LogRegLearner(lambda_=0.025)
classifier = learner(X, y)
draw_decision(X, y, classifier, 0, 1, grid, "Lambda 0.025", 2)

learner = LogRegLearner(lambda_=0.5)
classifier = learner(X, y)
draw_decision(X, y, classifier, 0, 1, grid, "Lambda 0.5", 3)
pylab.show()
# NALOGA 3

la = 0.64
no_outputs = 10
coef = 2

#test_cv
print("Rezultati z uporabo funkcije test_cv:")
for j in range(no_outputs):
    learner = LogRegLearner(lambda_=la)
    res = test_cv(learner, X, y)
    print("Lambda: ", la, " Tocnost:", CA(y, res))
    la /= coef
#test_learning
print()
print("Rezultati z uporabo funkcije test_learning:")
la = 0.64
for j in range(no_outputs):
    learner = LogRegLearner(lambda_=la)
    res = test_learning(learner, X, y)
    print("Lambda: ", la, " Tocnost:", CA(y, res))
    la /= coef


# NALOGA 4

# branje podatkov
with open('slike.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader)
    next(reader)
    next(reader)
    data = [d[:-4] for d in reader]
# pretvorimo string v float -> vsaka vrstica vsebuje 2048 atributov pridobljenih s pomocjo orange-a
X = [d[:-2] for d in data]
X = [list(map(float, d)) for d in X]
# mize preslikamo v vrednost 1 stole v vrednost 0
y = [1 if d[-2] == 'mize' else 0 for d in data]
imena_slik = [d[-1] for d in data]
print()
la = 0.64
print("Rezultati na lastnih slikah miz oz stolov z uporabo funkcije test_cv:")
for j in range(10):
    learner = LogRegLearner(lambda_=la)
    res = test_cv(learner, X, y)
    print("Lambda: ", la, " Tocnost:", CA(y, res))
    napacne = napacne_slike(y, res, imena_slik)
    print("Napacne slike: " + ", ".join(napacne))
    la /= coef
