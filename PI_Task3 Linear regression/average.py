import csv
import gzip

import lpputils


class AverageTripLearner(object):
    def __call__(self, data):
        delays = [lpputils.tsdiff(d[-1], d[-3]) for d in data]
        mean = sum(delays) / len(delays)

        return AverageTripClassifier(mean)


class AverageTripClassifier(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, x):
        # does not use the input example at all, because
        # in this case the prediction is always the same
        return self.mean


if __name__ == "__main__":

    f = gzip.open("train_pred.csv.gz", "rt")
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    data = [d for d in reader]

    l = AverageTripLearner()
    c = l(data)


    f = gzip.open("test_pred.csv.gz", "rt")
    reader = csv.reader(f, delimiter="\t")
    next(reader)  # skip legend

    fo = open("naloga3.txt", "wt")
    for l in reader:
        print(l[-3])
        fo.write(lpputils.tsadd(l[-3], c(l)) + "\n")
    fo.close()
