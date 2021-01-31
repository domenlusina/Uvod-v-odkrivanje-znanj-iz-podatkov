import csv
import gzip

import numpy

import linear
import lpputils


def weekday(x):
    if x.weekday() > 4:
        return 0
    else:
        return 1

#TODO popoldne
def getAttributes(od):
    x = [weekday(od), od.weekday(), od.second, od.minute, od.hour, od.day, od.month]
    return x


if __name__ == "__main__":
    f = gzip.open("train_pred.csv.gz", "rt")
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    # ['Registration', 'Driver ID', 'Route', 'Route Direction', 'Route description', 'First station', 'Departure time', 'Last station', 'Arrival time']

    data = [d for d in reader]
    noLines = len(data)

    Y = numpy.zeros(noLines)
    X = numpy.zeros([noLines, 7])
    for i, line in enumerate(data):
        Y[i] = lpputils.tsdiff(line[-1], line[-3])  # določimo čas vožnje
        odhod = lpputils.parsedate(line[-1])
        X[i] = getAttributes(odhod)

    lr = linear.LinearLearner(lambda_=1.)
    napovednik = lr(X, Y)

    f = gzip.open("test_pred.csv.gz", "rt")
    test = csv.reader(f, delimiter="\t")
    next(reader)  # skip legend



    fo = open("naloga3.txt", "wt")
    for l in test:
        odhod = lpputils.parsedate(l[-3])
        nov_primer = numpy.array(getAttributes(odhod))
        #print(nov_primer)
        c = napovednik(nov_primer)
        fo.write(lpputils.tsadd(l[-3], c) + "\n")
    fo.close()
