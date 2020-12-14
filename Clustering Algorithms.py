import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

def pairwiseL2(X, y=None):
    """
    Calculate l2 norm matrix (D) among X and y;

    X's shape is (n*m), y's shape is (k*m),
    and the D's shape is (n*k);
    Dij denotes the distance between Xi and yj;
    If y is None, then y will be X;

    :param X: ndarray (n*m)
    :param y: ndarray (k*m)
    :return: l2 norm matrix (n*k)
    """
    y = X if y is None else y

    return np.sqrt(np.abs(np.sum(X ** 2, axis=1).reshape((-1, 1)) \
                          + np.sum(np.transpose(y) ** 2, axis=0) - 2 * np.dot(X, np.transpose(y))))


def pairwiseL2Min(X, y):
    """
    Minimum values among X and y

    :param X: ndarray (n*m)
    :param y: ndarray (k*m)
    :return: minimum l2 norm matrix (n)
    """
    return np.min(pairwiseL2(X, y), axis=1)


def pairwiseL2ArgMin(X, y):
    """
    Index of minimum values among X and y

    :param X: ndarray (n*m)
    :param y: ndarray (k*m)
    :return: index of minimum l2 norm matrix (n)
    """
    return np.argmin(pairwiseL2(X, y), axis=1)


class Clustering:
    def __init__(self, nClusters, maxIter=600, tol=1e-4):
        self.nClusters = nClusters
        self.maxIter = maxIter
        self.iterCount = 0
        self.tol = tol

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X):
        raise NotImplementedError

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)


class KMeans(Clustering):
    def __init__(self, nClusters, nInit=10, maxIter=600, tol=1e-4):
        super(KMeans, self).__init__(nClusters=nClusters,
                                     maxIter=maxIter,
                                     tol=tol)
        self.centers = None
        self.nInit = nInit

    def initCenters(self, X):
        """
        Apply KMeans++ to initialize cluster centers.
        (http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)

        Init the first center randomly and
        iterate the following step until find all centers.

        Assign possibilities to all samples,
        which can be determined by the square of the distance
        from the nearest and selected centers.

        Weighted randomly select the next center
        according to possibilities assigned in the prev step.

        :param X: X
        :return: initialized centers
        """
        instances, features = X.shape
        centers = np.zeros([self.nClusters, features])
        # init the first center
        index = np.random.randint(0, instances - 1)
        centers[0] = X[index]

        # init rest of them according to distance
        for k in range(1, self.nClusters):
            D = pairwiseL2Min(X, centers[0:k, :])
            probability = (D**2) / (np.sum(D**2))
            centers[k] = X[np.argmax(probability)]

        return centers


    def kMeansSingleEM(self, X):
        """
        Run KMeans one time.

        Convergence condition: the square of the change of
        centers are less then tol

        :param X: X
        :return: (cluster centers, iteration count)
        """
        cluster_centers = self.initCenters(X)
        iteration = 0
        keep_going = True
        while keep_going:
            index_minimum = pairwiseL2ArgMin(X, cluster_centers)
            initial_centers = cluster_centers
            for k in range(self.nClusters):
                cluster_centers[k] = np.mean(X[index_minimum == k], axis=0)
            change_centres = cluster_centers - initial_centers
            if np.sum(change_centres ** 2) < self.tol:
                break
            iteration += 1
            if iteration > self.maxIter:
                keep_going = False

        return cluster_centers, iteration

    def score(self, X, centers=None):
        """
        Calculate distortion score of X and given centers.
        Distortion score = sum(square(||xi - nearest center||))

        If centers is None, centers will be self.centers

        :param X: X
        :param centers: centers
        :return: distortion score
        """
        centers = self.centers if centers is None else centers

        return np.sum((pairwiseL2Min(X, centers)**2))


    def fit(self, X, y=None):
        """
        Run KMeans nInit times,
        and pick the best one according to distortion score.

        Record the best cluster centers in self.centers,
        and save the corresponding iteration times in self.iterCount

        :param X: X
        :param y: None
        :return: self
        """
        instances, features = X.shape
        centers = np.zeros([self.nInit, self.nClusters, features])
        iterCount = np.zeros([self.nInit])
        scores = np.zeros([self.nInit])

        for i in range(self.nInit):
            centers[i, :, :], iterCount[i] = self.kMeansSingleEM(X)
            scores[i] = self.score(X, centers[i, :, :])

        self.centers = centers[np.argmin(scores), :, :]
        self.iterCount = iterCount[np.argmin(scores)]
        return self


    def predict(self, X):
        """
        Predict clusters of given X

        :param X: X
        :return: clusters (nX)
        """

        return pairwiseL2ArgMin(X, self.centers)


class GMM(Clustering):
    def __init__(self, nClusters, maxIter=600, tol=1e-4):
        super(GMM, self).__init__(nClusters=nClusters, maxIter=maxIter, tol=tol)
        self.priors = None
        self.means = None
        self.covariances = None

    def initPosteriors(self, X):
        """
        Use KMeans to initialize posteriors.

        :param X: X
        :return: posteriors (nX, nClusters)
        """
        kmeans_model = KMeans(self.nClusters)
        kmeans_model.fit(X)
        clusters = kmeans_model.predict(X)

        Posterior = np.zeros((clusters.size, clusters.max()+1), dtype=int)
        Posterior[np.arange(clusters.size), clusters] = 1

        return Posterior


    def initialization(self, X):
        """
        Initialize posteriors and prior, mean and covariances.
        It's better to implement estimateGaussianParams first.

        :param X: X
        """
        posteriors = self.initPosteriors(X)
        self.priors, self.means, self.covariances = self.estimateGaussianParams(X, posteriors)


    def estimateGaussianParams(self, X, posteriors):
        """
        M step

        Estimate priors, means and covariance
        according to given X and posteriors.

        :param X: X
        :param posteriors: (nX, nClusters)
        :return: (priors, means, covariance)
        """
        nX, dim = X.shape

        priors = np.mean(posteriors, axis=0)

        mean = np.array([self.nClusters, dim])
        mean = np.sum(X * np.transpose((np.array([posteriors]*dim))), axis=1)\
               / np.transpose((np.array([np.sum(posteriors, axis=0)] *dim)))

        cov = np.zeros([self.nClusters, dim, dim])
        for i in range(nX):
            cov[posteriors[i, :] == 1, :, :] += np.dot(np.transpose((X[i] - mean[posteriors[i, :] == 1, :]).reshape(1, -1)), (X[i] - mean[posteriors[i, :] == 1, :]).reshape(1, -1))

        cov /= np.transpose((np.array([np.array([np.sum(posteriors, axis=0)] * dim)] * dim)))

        return priors, mean, cov


    def estimateScorePosteriors(self, X):
        """
        E step

        Estimate score and posteriors according to X, priors, means, and covariances.

        :param X: X
        :return: (score, posteriors)
        """
        instances, features = X.shape
        log_joint = np.zeros([instances, self.nClusters])

        for k in range(self.nClusters):
            log_joint[:, k] = multivariate_normal(self.means[k, :], self.covariances[k, :, :]).logpdf(X) + np.log(self.priors[k])
        posterior = np.transpose(np.transpose(log_joint) - logsumexp(log_joint, axis=1))
        score = np.average(logsumexp(log_joint, axis=1))
        t = np.exp(posterior)
        post = np.zeros([X.shape[0], self.nClusters])
        post[range(X.shape[0]), np.argmax(t, axis=1)] = 1

        return score, post


    def fit(self, X, y=None):
        """
        Train GMM using EM algorithm.
        If maxIter is None, it should iterate forever until it converges

        :param X: X
        :param y: None
        :return: self
        """
        # Invoke initialization to apply KMeans initialize parameters
        self.initialization(X)

        # Iterate maxIter times to perform EM process until it converges
        iteration = 0
        keep_going = True

        while keep_going:
            score, posteriors = self.estimateScorePosteriors(X)
            self.priors, self.means, self.covariances = self.estimateGaussianParams(X, posteriors)
            iteration += 1
            if self.maxIter is not None and iteration >= self.maxIter:
                keep_going = False

        return self


    def predict(self, X):
        """
        Predict X's clusters

        :param X: X
        :return: clusters (nX)
        """
        score, posteriors = self.estimateScorePosteriors(X)
        clusters = np.where(posteriors == 1)[1]
        return clusters

    def score(self, X):
        """
        Predict the parameter's score by given X

        :param X: X
        :return: score
        """
        score, posteriors = self.estimateScorePosteriors(X)

        return score


class PCA:
    def __init__(self, nComponents=2):
        self.nComponents = nComponents

    def fit_transform(self, X):
        """
        Reduce the dimension of given X and nComponents

        :param X: X
        :return: transformed X (nX, nComponents)
        """
        # Calculate covariance matrix
        cenX = X - np.mean(X, axis=0)
        covCenX = np.cov(np.transpose(cenX))

        # Calculate eigen values and eigen vectors,
        # select eigen vectors corresponding to
        # top nComponents eigen values,
        # and return the transformed X.
        eigVal, eigVec = np.linalg.eig(covCenX)
        top_eigVec = eigVec[np.flip(np.argsort(eigVal))[0:self.nComponents]]
        return np.dot(X, np.transpose(top_eigVec))
