import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Define Thevenin model function
def thevenin_model(t, Voc, R0, C, I):
    Vt = Voc - R0 * I
    return Vt * np.exp(-t / (R0 * C)) + R0 * I

# Define function to calculate Jacobian matrix
def calc_jacobian(f, x, delta):
    n = len(x)
    J = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        x1 = x.copy()
        x1[i] += delta
        J[:, i] = (f(x1) - fx) / delta
    return J

# Define extended Kalman filter function
def ekf(battery_data):
    # Initialize parameters and variables
    Voc_guess = battery_data[0]
    R0_guess = 0.1
    C_guess = 1000
    I_guess = 0
    x = np.array([Voc_guess, R0_guess, C_guess, I_guess])
    P = np.eye(4)
    delta = 1e-5
    Q = np.diag([0.1, 0.01, 0.01, 0.001])
    R = 0.05 ** 2
    time_to_discharge = np.nan

    # Define state transition function
    def f(x):
        Voc, R0, C, I = x
        return np.array([Voc - R0 * I, R0, C, I])

    # Loop through battery data and update state estimate
    for i in range(1, len(battery_data)):
        # Predict next state and Jacobian matrix
        x_predicted = thevenin_model(1, *x)
        J = calc_jacobian(thevenin_model, x, delta)

        # Update error covariance matrix
        P_predicted = np.matmul(np.matmul(J, P), J.T) + Q

        # Update measurement and measurement function
        z = battery_data[i]
        h = x_predicted

        # Calculate Kalman gain and update state estimate and error covariance matrix
        y = z - h
        S = np.matmul(np.matmul(J, P_predicted), J.T) + R
        K = np.matmul(np.matmul(P_predicted, J.T), np.linalg.inv(S))
        x = x_predicted + np.matmul(K, y)
        P = np.matmul(np.eye(4) - np.matmul(K, J), P_predicted)

        # Check if battery is discharged
        if x[0] - x[1] * x[3] <= 0 and np.isnan(time_to_discharge):
            time_to_discharge = i

    return time_to_discharge
