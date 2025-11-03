
import numpy as np
from scipy import sparse as ssparse
import sklearn.linear_model as sklinear


class reduce_feature_matrix(object):
    """
    Performs dimensionality reduction on the feature matrix.
    Currently only supports reduced rank regression.

    Parameters
    ----------------
    X : (numpy ndarray)
        feature matrix with shape (num_samples, num_features)
    Y : (numpy ndarray)
        target matrix with shape (num_samples, num_target_variables)
    rank : int
        number of dimensions to keep
    method : str
        how to implement the dimensionality reduction
        'reduced-rank' : perform reduced rank regression using code from  Chris Rayner (2015)
        'reduced-rank-steinmetz': same implementation as found in Stienmetz (2019)
        see: https://bit.ly/34jQf4Z

    # TODO: perhaps take the PB matrix in fit, then trasfnrom will just be PBX
    """

    def __init__(self, rank=5, reg=0, method='reduced-rank'):
        self.rank = rank
        self.reg = reg
        self.method = method
        self.transformer_matrix = None  # needs to be fitted.

    def fit(self, X, Y):

        assert self.method in ['reduced-rank', 'reduced-rank-steinmetz'], print('Unknown method specified.')
        if self.method == 'reduced-rank':
            # weird implementation, but get the same results as Kush's reduced rank code
            CXX = np.dot(X.T, X) + self.reg * ssparse.eye(np.size(X, 1))
            CXY = np.dot(X.T, Y)
            _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))

            W = V[0:self.rank, :].T
            A = np.dot(np.linalg.pinv(CXX), np.dot(CXY, W)).T
            self.transformer_matrix = A.T  # same as B in Steinmetz 2019

            # XA = np.dot(X, A.T)  # same as PB in Steinmetz 2019
            # reduced_X = XA

        elif self.method == 'reduced-rank-steinmetz':

            CYX = np.dot(Y.T, X)
            CXX = np.dot(X.T, X) + self.reg * ssparse.eye(np.size(X, 1))
            CXXMH = np.sqrt(CXX)
            M = np.dot(CYX, CXXMH)

            U, S, V = np.linalg.svd(M)
            B = np.dot(CXXMH, V)
            # _A = np.dot(U, S)

            reduced_B = B[:, :self.rank]
            self.transformer_matrix = reduced_B
            # reduced_X = np.dot(X, reduced_B)

        return self

    def transform(self, X):

        reduced_X = np.dot(X, self.transformer_matrix)

        return reduced_X

    def fit_transform(self, X, Y):

        if self.method == 'reduced-rank':
            # weird implementation, but get the same results as Kush's reduced rank code
            CXX = np.dot(X.T, X) + self.reg * ssparse.eye(np.size(X, 1))
            CXY = np.dot(X.T, Y)
            _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))

            W = V[0:self.rank, :].T
            A = np.dot(np.linalg.pinv(CXX), np.dot(CXY, W)).T
            XA = np.dot(X, A.T)  # same as PB in Steinmetz 2019
            reduced_X = XA

        elif self.method == 'reduced-rank-steinmetz':

            CYX = np.dot(Y.T, X)
            CXX = np.dot(X.T, X) + self.reg * ssparse.eye(np.size(X, 1))
            CXXMH = np.sqrt(CXX)
            M = np.dot(CYX, CXXMH)

            U, S, V = np.linalg.svd(M)
            B = np.dot(CXXMH, V)
            # _A = np.dot(U, S)

            reduced_B = B[:, :self.rank]
            reduced_X = np.dot(X, reduced_B)

        return reduced_X

    def get_params(self, deep=True):
        return {'rank': self.rank, 'reg': self.reg}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class ReducedRankRegressor(object):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    Implemented by Chris Rayner (2015): dchrisrayner AT gmail DOT com.
    With some extensions to by Tim Sit to calculate the error.
    Also restructured in a way that fits better with sklearn model objects.
    """

    def __init__(self, rank=10, reg=0, regressor=None, alpha=0, l1_ratio=1):
        self.reg = reg
        self.rank = rank
        self.regressor = regressor
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __str__(self):
        return 'Reduced Rank Regressor (rank = {})'.format(self.rank)

    def fit(self, X, Y):

        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))

        CXX = np.dot(X.T, X) + self.reg * ssparse.eye(np.size(X, 1)) # I guess that used to be some sort of regularisation
        CXY = np.dot(X.T, Y)
        # _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))
        matrix_to_do_SVD = np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY))

        if np.isnan(matrix_to_do_SVD).any():
            # TODO: perhaps the better option is to drop those feature columns...
            print('NaNs detected, replacing them with zeros')
            matrix_to_do_SVD[np.isnan(matrix_to_do_SVD)] = 0

        _U, _S, V = np.linalg.svd(matrix_to_do_SVD)
        self.W = V[0:self.rank, :].T
        self.A = np.dot(np.linalg.pinv(CXX), np.dot(CXY, self.W)).T

        if self.regressor == 'Ridge':
            self.XA = np.asarray(np.dot(X, self.A.T))  # same as PB in Steinmetz 2019
            self.regressor_model = sklinear.Ridge(alpha=self.alpha, fit_intercept=False, solver='auto')
            self.regressor_model.fit(X=self.XA, y=Y)
        elif self.regressor == 'ElasticNet':
            self.XA = np.dot(X, self.A.T)  # same as PB in Steinmetz 2019
            self.regressor_model = sklinear.ElasticNet(fit_intercept=False, alpha=self.alpha,
                                                       l1_ratio=self.l1_ratio)
            self.regressor_model.fit(X=self.XA, y=Y)

        # recalculate the kernels
        self.coef_ = np.dot(self.regressor_model.coef_,self.A)

        return self

    def predict(self, X):
        """Predict Y from X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))

        if self.regressor is None:
            Y_hat = np.dot(X, np.dot(self.A.T, self.W.T))
        else:
            XA = np.asarray(np.dot(X, self.A.T))  # multiply new data with the A (ie. B) matrix that was learnt
            Y_hat = self.regressor_model.predict(XA)

        return Y_hat

    def score(self, X, Y):

        Y_hat = self.predict(X)
        u = np.sum(np.power(Y - Y_hat, 2))  # residual sum of squares
        v = np.sum(np.power(Y - np.mean(Y), 2))  # total sum of squares
        R_2 = 1 - u / v

        return Y, R_2

    def get_params(self, deep=True):
        return {'rank': self.rank, 'reg': self.reg}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self




