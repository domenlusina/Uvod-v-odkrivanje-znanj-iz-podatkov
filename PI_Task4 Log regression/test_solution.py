import unittest
import numpy
import solution  # your code with logistic regression and evaluation should be in solution.py


def data1():
    X = numpy.array([[5.0, 3.6, 1.4, 0.2],
                     [5.4, 3.9, 1.7, 0.4],
                     [4.6, 3.4, 1.4, 0.3],
                     [5.0, 3.4, 1.5, 0.2],
                     [5.6, 2.9, 3.6, 1.3],
                     [6.7, 3.1, 4.4, 1.4],
                     [5.6, 3.0, 4.5, 1.5],
                     [5.8, 2.7, 4.1, 1.0]])
    y = numpy.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def data2():
    X, y = data1()
    X = X[:6]
    y = y[:6]
    return X[:6], y[:6]


class TestLogisticRegression(unittest.TestCase):

    def test_h(self):
        X, y = data1()
        self.assertAlmostEqual(solution.h(X[0], numpy.array([0, 0, 0, 0])), 0.5)
        self.assertAlmostEqual(solution.h(X[0], numpy.array([0.1, 0.1, 0.1, 0.1])), 0.73497259)
        self.assertAlmostEqual(solution.h(X[0], numpy.array([0.1, 0.2, 0.1, 0.7])), 0.817574476)

    def test_cost_noreg(self):
        X, y = data1()
        self.assertAlmostEqual(solution.cost(numpy.array([0, 0, 0, 0]), X, y, 0.0), 0.69314718)
        self.assertAlmostEqual(solution.cost(numpy.array([0.1, -0.1, 0.1, 0.1]), X, y, 0.0), 0.61189876)
        self.assertAlmostEqual(solution.cost(numpy.array([1, 1, 1, 1]), X, y, 0.0), 5.17501926)
        self.assertAlmostEqual(solution.cost(numpy.array([-1.06, -5.39, 7.64, 3.79]), X, y, 0.0), 6.152e-06)
        X, y = data2()
        self.assertAlmostEqual(solution.cost(numpy.array([-1.06, -5.39, 7.64, 3.79]), X, y, 0.0), 8.108543015770552e-06)

    def test_grad_noreg(self):
        X, y = data1()
        numpy.testing.assert_allclose(solution.grad(numpy.array([0, 0, 0, 0]), X, y, 0.0),
            [-0.2313, 0.1625, -0.6625, -0.2563], rtol=1e-3)
        numpy.testing.assert_allclose(solution.grad(numpy.array([0.1, -0.1, 0.1, 0.1]), X, y, 0.0),
            [0.5610, 0.5969, -0.1870, -0.1151], rtol=1e-3)
        numpy.testing.assert_allclose(solution.grad(numpy.array([-1.06, -5.39, 7.64, 3.79]), X, y, 0.0),
            [0., 0., 0., 0.], atol=1e-3)
        X, y = data2()
        numpy.testing.assert_allclose(solution.grad(numpy.array([0.1, -0.1, 0.1, 0.1]), X, y, 0.0),
            [1.3211, 1.0822, 0.1827, -0.0282], rtol=1e-3)

    def test_regularization(self):
        X, y = data1()

        absth0 = numpy.abs(solution.LogRegLearner(0.0)(X, y).th[1:])
        absth10 = numpy.abs(solution.LogRegLearner(10.0)(X, y).th[1:])
        absth100 = numpy.abs(solution.LogRegLearner(100.0)(X, y).th[1:])

        self.assertTrue(numpy.all(absth0 > absth10))
        self.assertTrue(numpy.all(absth10 > absth100))

    def test_z_grad_cost_compatible(self):
        X, y = data1()

        def diff_numberic_analitic_grad(theta, lambda_, analitic_grad):
            thw = theta.copy()
            eps = 10**-4
            vec = numpy.zeros(len(thw))
            for i in range(len(vec)):
                thw[i] += eps
                high = solution.cost(thw, X, y, lambda_)
                thw[i] -= 2*eps
                low = solution.cost(thw, X, y, lambda_)
                thw[i] += eps  # go back
                vec[i] = (high-low)/(2*eps)
            return numpy.max(numpy.abs(analitic_grad-vec))

        thetas = [[-0.231, 0.162, -0.662, -0.256],
                  [0.561, 0.597, -0.187, -0.115],
                  [-1.06, -5.39, 7.64, 3.79]]
    
        for theta in thetas:
            for lambda_ in [0.0, 1.0, 100.0, 0.0001]:
                theta = numpy.array(theta)
                an_grad = solution.grad(theta, X, y, lambda_)
                self.assertAlmostEqual(diff_numberic_analitic_grad(theta, lambda_, an_grad), 0.0, places=6)


class DummyCVLearner:
    """ For CV testing """
    def __call__(self, X, y):
        return DummyCVClassifier(ldata=X)


class YouTestOnTrainingData(Exception): pass


class FoldsNotEqualSize(Exception): pass


class NotAllTested(Exception): pass


class MixedOrder(Exception): pass


class DummyCVClassifier:
    def __init__(self, ldata):
        self.ldata = list(map(list, ldata))

    def __call__(self, x):
        if list(x) in self.ldata:
            raise YouTestOnTrainingData()
        else:
            return [sum(x), len(self.ldata)]


class TestEvaluation(unittest.TestCase):

    def test_ca(self):
        X, y = data1()
        self.assertAlmostEqual(solution.CA(y, [[1, 0]]*len(y)), 0.5)
        self.assertAlmostEqual(solution.CA(y, [[0.5, 1]]*len(y)), 0.5)
        self.assertAlmostEqual(solution.CA(y[:4], [[0.4, 0.6]]*len(y[:4])), 0.0)
        self.assertAlmostEqual(solution.CA(y[4:], [[0.4, 0.6]]*len(y[4:])), 1.0)
        self.assertAlmostEqual(solution.CA(y[:6], [[0.4, 0.6]]*len(y[:6])), 2/6.)
        self.assertAlmostEqual(solution.CA(y, [[0, 1], [0.2, 0.8], [0, 1], [0, 1], [1, 0], [0.6, 0.4], [1, 0], [1, 0]]), 0.0)

    def test_logreg_noreg_learning_ca(self):
        X, y = data1()
        logreg = solution.LogRegLearner(lambda_=0)
        pred = solution.test_learning(logreg, X, y)
        ca = solution.CA(y, pred)
        self.assertAlmostEqual(ca, 1.)

    def test_cv(self):
        for X, y in [data1(), data2()]:
            pred = solution.test_cv(DummyCVLearner(), X, y, k=4)

            if len(y) == 8:
                # training data should have 6 instances
                self.assertEqual(pred[0][1], 6)

            if len(set([a for _, a in pred])) != 1 and len(y) % 4 == 0:
                raise FoldsNotEqualSize()

            signatures = [a for a, _ in pred]
            if len(set(signatures)) != len(y):
                raise NotAllTested()

            if signatures != list(map(lambda x: sum(list(x)), X)):
                raise MixedOrder()

    def test_cv_shuffled(self):
        """Do not take folds in order - shuffle because data is frequently clustered """
        _, y = data_iris()
        X = numpy.array([[i] for i in range(100)])
        pred = solution.test_cv(DummyShuffleLearner(), X, y, k=4)


class TooManyConsecutiveInstances(Exception): pass


class DummyShuffleLearner:
    """ For CV testing """
    def __call__(self, X, y):
        X = X.T
        notinorder = len(numpy.flatnonzero(numpy.abs(numpy.diff(X)) - 1))
        if notinorder < 8:
            raise TooManyConsecutiveInstances
        return lambda x: [0.5, 0.5]


def data_iris():
    irisX = numpy.array([[5.1, 3.5, 1.4, 0.2],
                         [4.9, 3., 1.4, 0.2],
                         [4.7, 3.2, 1.3, 0.2],
                         [4.6, 3.1, 1.5, 0.2],
                         [5., 3.6, 1.4, 0.2],
                         [5.4, 3.9, 1.7, 0.4],
                         [4.6, 3.4, 1.4, 0.3],
                         [5., 3.4, 1.5, 0.2],
                         [4.4, 2.9, 1.4, 0.2],
                         [4.9, 3.1, 1.5, 0.1],
                         [5.4, 3.7, 1.5, 0.2],
                         [4.8, 3.4, 1.6, 0.2],
                         [4.8, 3., 1.4, 0.1],
                         [4.3, 3., 1.1, 0.1],
                         [5.8, 4., 1.2, 0.2],
                         [5.7, 4.4, 1.5, 0.4],
                         [5.4, 3.9, 1.3, 0.4],
                         [5.1, 3.5, 1.4, 0.3],
                         [5.7, 3.8, 1.7, 0.3],
                         [5.1, 3.8, 1.5, 0.3],
                         [5.4, 3.4, 1.7, 0.2],
                         [5.1, 3.7, 1.5, 0.4],
                         [4.6, 3.6, 1., 0.2],
                         [5.1, 3.3, 1.7, 0.5],
                         [4.8, 3.4, 1.9, 0.2],
                         [5., 3., 1.6, 0.2],
                         [5., 3.4, 1.6, 0.4],
                         [5.2, 3.5, 1.5, 0.2],
                         [5.2, 3.4, 1.4, 0.2],
                         [4.7, 3.2, 1.6, 0.2],
                         [4.8, 3.1, 1.6, 0.2],
                         [5.4, 3.4, 1.5, 0.4],
                         [5.2, 4.1, 1.5, 0.1],
                         [5.5, 4.2, 1.4, 0.2],
                         [4.9, 3.1, 1.5, 0.1],
                         [5., 3.2, 1.2, 0.2],
                         [5.5, 3.5, 1.3, 0.2],
                         [4.9, 3.1, 1.5, 0.1],
                         [4.4, 3., 1.3, 0.2],
                         [5.1, 3.4, 1.5, 0.2],
                         [5., 3.5, 1.3, 0.3],
                         [4.5, 2.3, 1.3, 0.3],
                         [4.4, 3.2, 1.3, 0.2],
                         [5., 3.5, 1.6, 0.6],
                         [5.1, 3.8, 1.9, 0.4],
                         [4.8, 3., 1.4, 0.3],
                         [5.1, 3.8, 1.6, 0.2],
                         [4.6, 3.2, 1.4, 0.2],
                         [5.3, 3.7, 1.5, 0.2],
                         [5., 3.3, 1.4, 0.2],
                         [7., 3.2, 4.7, 1.4],
                         [6.4, 3.2, 4.5, 1.5],
                         [6.9, 3.1, 4.9, 1.5],
                         [5.5, 2.3, 4., 1.3],
                         [6.5, 2.8, 4.6, 1.5],
                         [5.7, 2.8, 4.5, 1.3],
                         [6.3, 3.3, 4.7, 1.6],
                         [4.9, 2.4, 3.3, 1.],
                         [6.6, 2.9, 4.6, 1.3],
                         [5.2, 2.7, 3.9, 1.4],
                         [5., 2., 3.5, 1.],
                         [5.9, 3., 4.2, 1.5],
                         [6., 2.2, 4., 1.],
                         [6.1, 2.9, 4.7, 1.4],
                         [5.6, 2.9, 3.6, 1.3],
                         [6.7, 3.1, 4.4, 1.4],
                         [5.6, 3., 4.5, 1.5],
                         [5.8, 2.7, 4.1, 1.],
                         [6.2, 2.2, 4.5, 1.5],
                         [5.6, 2.5, 3.9, 1.1],
                         [5.9, 3.2, 4.8, 1.8],
                         [6.1, 2.8, 4., 1.3],
                         [6.3, 2.5, 4.9, 1.5],
                         [6.1, 2.8, 4.7, 1.2],
                         [6.4, 2.9, 4.3, 1.3],
                         [6.6, 3., 4.4, 1.4],
                         [6.8, 2.8, 4.8, 1.4],
                         [6.7, 3., 5., 1.7],
                         [6., 2.9, 4.5, 1.5],
                         [5.7, 2.6, 3.5, 1.],
                         [5.5, 2.4, 3.8, 1.1],
                         [5.5, 2.4, 3.7, 1.],
                         [5.8, 2.7, 3.9, 1.2],
                         [6., 2.7, 5.1, 1.6],
                         [5.4, 3., 4.5, 1.5],
                         [6., 3.4, 4.5, 1.6],
                         [6.7, 3.1, 4.7, 1.5],
                         [6.3, 2.3, 4.4, 1.3],
                         [5.6, 3., 4.1, 1.3],
                         [5.5, 2.5, 4., 1.3],
                         [5.5, 2.6, 4.4, 1.2],
                         [6.1, 3., 4.6, 1.4],
                         [5.8, 2.6, 4., 1.2],
                         [5., 2.3, 3.3, 1.],
                         [5.6, 2.7, 4.2, 1.3],
                         [5.7, 3., 4.2, 1.2],
                         [5.7, 2.9, 4.2, 1.3],
                         [6.2, 2.9, 4.3, 1.3],
                         [5.1, 2.5, 3., 1.1],
                         [5.7, 2.8, 4.1, 1.3]])

    irisy = numpy.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    return irisX, irisy


class TestAdditional(unittest.TestCase):

    def test_auc(self):
        X, y = data_iris()
        pred = [(-x[0], x[0]) for x in X]

        # scores for directROC curve, from Mann-Whitney test, or from direct (tie-corrected) definition
        correctaucs = numpy.array([0.9326,  # 100
                                   0.94133333,  # 80
                                   0.899])  # 60

        for num, correct in zip([100, 80, 60], correctaucs):
            real = y[:num]
            preds = pred[:num]

            auc = solution.AUC(real, preds)
            self.assertAlmostEqual(auc, correct, places=3)


if __name__ == '__main__':
    unittest.main()
