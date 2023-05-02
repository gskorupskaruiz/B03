# import all packages and set plots to be embedded inline
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from pyDOE import lhs
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

k = 50
Bat_1 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0005').to_numpy()[:k,:]
Bat_2 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0018').to_numpy()[:k,:]
Bat_3 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0007').to_numpy()[:k,:]
Bat_4 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0006').to_numpy()[:k,:]

Bat = Bat_1
x_columns = [0,1,2,3,4,7] # voltage, temp, capacity
x_train, x_test, y_train, y_test = train_test_split(Bat[:,x_columns], Bat[:,8], test_size=0.2, random_state=None, shuffle=False)




class GaussianProcess:
    """A Gaussian Process class for creating and exploiting
    a Gaussian Process model"""

    def __init__(self, n_restarts, optimizer):
        """Initialize a Gaussian Process model

        Input
        ------
        n_restarts: number of restarts of the local optimizer
        optimizer: algorithm of local optimization"""

        self.n_restarts = n_restarts
        self.optimizer = optimizer

    '''
    def Corr(self, X1, X2, ls_M, nu_M, ls_ess, periodicity_ess, const):
        """Construct the correlation matrix between X1 and X2

        Input
        -----
        X1, X2: 2D arrays, (n_samples, n_features)
        theta: array, correlation legnths for different dimensions

        Output
        ------
        K: the correlation matrix
        """
        
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            d = np.sqrt(np.sum(theta * (X1[i, :] - X2) ** 2, axis=1))
            rbf  = np.exp(-d**2/(2*ls_M**2))
            ess = np.exp(- 2*np.sin(np.pi * d /periodicity_ess) / ls_ess**2        )
            constant  = const
            K[i, :] = rbf + ess + constant      #  np.exp(-np.sum(theta * (X1[i, :] - X2) ** 2, axis=1))
        
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            d = np.sqrt(np.sum(theta * (X1[i, :] - X2) ** 2, axis=1))
            rbf  = np.exp(-d**2/(2*ls_M**2))
            ess = np.exp(- 2*np.sin(np.pi * d /periodicity_ess) / ls_ess**2        )
            constant  = const
            K[i, :] = rbf + ess + constant      #  np.exp(-np.sum(theta * (X1[i, :] - X2) ** 2, axis=1))

        return K
    '''

    def Corr(self, X1, X2, theta):
        """Construct the correlation matrix between X1 and X2

        Input
        -----
        X1, X2: 2D arrays, (n_samples, n_features)
        theta: array, correlation legnths for different dimensions

        Output
        ------
        K: the correlation matrix
        """
        '''
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            d = np.sqrt(np.sum(theta * (X1[i, :] - X2) ** 2, axis=1))
            rbf  = np.exp(-d**2/(2*ls_M**2))
            ess = np.exp(- 2*np.sin(np.pi * d /periodicity_ess) / ls_ess**2        )
            constant  = const
            K[i, :] = rbf + ess + constant      #  np.exp(-np.sum(theta * (X1[i, :] - X2) ** 2, axis=1))
        '''
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            K[i, :] = np.exp(-np.sum(theta * (X1[i, :] - X2) ** 2, axis=1))

        return K


    def Neglikelihood(self, theta):
        """Negative likelihood function

        Input
        -----
        theta: array, logarithm of the correlation legnths for different dimensions

        Output
        ------
        LnLike: likelihood value"""

        theta = 10 ** theta  # Correlation length
        n = self.X.shape[0]  # Number of training instances
        one = np.ones((n, 1))  # Vector of ones

        # Construct correlation matrix
        K = self.Corr(self.X, self.X, theta) + np.eye(n) * 1e-10
        inv_K = np.linalg.inv(K)  # Inverse of correlation matrix

        # Mean estimation
        mu = (one.T @ inv_K @ self.y) / (one.T @ inv_K @ one)

        # Variance estimation
        SigmaSqr = (self.y - mu * one).T @ inv_K @ (self.y - mu * one) / n

        # Compute log-likelihood
        DetK = np.linalg.det(K)
        LnLike = -(n / 2) * np.log(SigmaSqr) - 0.5 * np.log(DetK)

        # Update attributes
        self.K, self.inv_K, self.mu, self.SigmaSqr = K, inv_K, mu, SigmaSqr

        return -LnLike.flatten()


    def fit(self, X, y):
        """GP model training

        Input
        -----
        X: 2D array of shape (n_samples, n_features)
        y: 2D array of shape (n_samples, 1)
        """

        self.X, self.y = X, y
        lb, ub = -3, 2

        # Generate random starting points (Latin Hypercube)
        lhd = lhs(self.X.shape[1], samples=self.n_restarts)

        # Scale random samples to the given bounds
        initial_points = (ub - lb) * lhd + lb

        # Create A Bounds instance for optimization
        bnds = Bounds(lb * np.ones(X.shape[1]), ub * np.ones(X.shape[1]))

        # Run local optimizer on all points
        opt_para = np.zeros((self.n_restarts, self.X.shape[1]))
        opt_func = np.zeros((self.n_restarts, 1))
        for i in range(self.n_restarts):
            res = minimize(self.Neglikelihood, initial_points[i, :], method=self.optimizer,
                           bounds=bnds)
            opt_para[i, :] = res.x
            opt_func[i, :] = res.fun

        # Locate the optimum results
        self.theta = opt_para[np.argmin(opt_func)]

        # Update attributes
        self.NegLnlike = self.Neglikelihood(self.theta)


    def predict(self, X_test):
        """GP model predicting

        Input
        -----
        X_test: test set, array of shape (n_samples, n_features)

        Output
        ------
        f: GP predictions
        SSqr: Prediction variances"""

        n = self.X.shape[0]
        one = np.ones((n, 1))

        # Construct correlation matrix between test and train data
        k = self.Corr(self.X, X_test, 10 ** self.theta)

        # Mean prediction
        f = self.mu + k.T @ self.inv_K @ (self.y - self.mu * one)

        # Variance prediction
        SSqr = self.SigmaSqr * (1 - np.diag(k.T @ self.inv_K @ k))

        return f.flatten(), SSqr.flatten()



    def score(self, X_test, y_test):
        """Calculate root mean squared error

        Input
        -----
        X_test: test set, array of shape (n_samples, n_features)
        y_test: test labels, array of shape (n_samples, )

        Output
        ------
        RMSE: the root mean square error"""

        y_pred, SSqr = self.predict(X_test)
        RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))

        return RMSE


def Test_1D(X):
    """1D Test Function"""

    y = (X * 6 - 2) ** 2 * np.sin(X * 12 - 4)

    return y

'''
# Training data
X_train = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1], ndmin=2).T
y_train = Test_1D(X_train)

# Testing data
X_test = np.linspace(0.0, 1, 100).reshape(-1, 1)
y_test = Test_1D(X_test)

# GP model training
GP = GaussianProcess(n_restarts=10, optimizer='L-BFGS-B')
GP.fit(X_train, y_train)

# GP model predicting
y_pred, y_pred_SSqr = GP.predict(X_test)


plt.plot(X_test, y_pred)
plt.plot(X_test, Test_1D(X_test))
plt.show()



def Test_2D(X):
    """2D Test Function"""

    y = (1 - X[:, 0]) ** 2 + 100 * (X[:, 1] - X[:, 0] ** 2) ** 2

    return y

# Training data
sample_num = 25
lb, ub = np.array([-2, -1]), np.array([2, 3])
X_train = (ub-lb)*lhs(2, samples=sample_num) + lb
y_train = Test_2D(X_train).reshape(-1,1)

# Test data
X1 = np.linspace(-2, 2, 20)
X2 = np.linspace(-1, 3, 20)
X1, X2 = np.meshgrid(X1, X2)
X_test = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
y_test = Test_2D(X_test)

# GP model training
pipe = Pipeline([('scaler', MinMaxScaler()),
         ('GP', GaussianProcess(n_restarts=10, optimizer='L-BFGS-B'))])
pipe.fit(X_train, y_train)

# GP model predicting
y_pred, y_pred_SSqr = pipe.predict(X_test)

# Accuracy score
pipe.score(X_test, y_test)
plt.plot(X_test, y_test)
plt.show()

'''

X1, X2 = np.meshgrid(X1, X2)
X_test = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))



pipe = Pipeline([('scaler', MinMaxScaler()),
         ('GP', GaussianProcess(n_restarts=10, optimizer='L-BFGS-B'))])
pipe.fit(x_train, y_train)
#GP = GaussianProcess(n_restarts=10, optimizer='L-BFGS-B')
#GP.fit(np.array(x_train,ndmin=2).T, y_train)

#y_pred_1 = GP.predict(x_test)[0]
y_pred_1 = pipe.predict(x_test)[0]

differences_1 = (y_pred_1[y_pred_1!=0] - y_test[y_pred_1!=0])/y_pred_1[y_pred_1!=0]*100


plt.plot(np.linspace(0,1,len(y_pred_1)), y_pred_1)
plt.plot(np.linspace(0,1,len(y_test)), y_test)

plt.xlabel('test instance')
plt.ylabel('test-truth % error ')
plt.show()
