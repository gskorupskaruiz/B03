import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from filterpy.kalman import ExtendedKalmanFilter

# Define Thevenin model function
def thevenin_model(t, Voc, R0, C, I):
    Vt = Voc - R0 * I
    return Vt * np.exp(-t / (R0 * C)) + R0 * I

# Define function to calculate Jacobian matrix
"""def calc_jacobian(f, x, delta):
    n = len(x)
    J = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        x1 = x.copy()
        x1[i] += delta
        J[:, i] = (f(x1) - fx) / delta
    return J"""

# Initialize the state variables
SoC = [100]  # Start with a fully charged battery
OCV = [V_thevenin + R_thevenin * current[0]]  # Assume the battery's fully charged voltage is the Thevenin voltage
x = np.array([SoC[0], OCV[0], V_thevenin, R_thevenin]).reshape(-1, 1)

# Define the state transition function
def state_transition(x, current, voltage, delta_t, capacity, R0, RC):
    SoC, OCV, V_thevenin, R_thevenin = x.ravel()
    I = current
    V = voltage

    # Update the Thevenin voltage and resistance
    V_thevenin_new = OCV - I * R_thevenin
    R_thevenin_new = R0 + RC / (delta_t + RC) * R_thevenin

    # Calculate SoC and OCV
    SoC_new = SoC - (I * delta_t) / capacity
    OCV_new = V_thevenin_new

    # Return the updated state vector
    x_new = np.array([SoC_new, OCV_new, V_thevenin_new, R_thevenin_new]).reshape(-1, 1)
    return x_new
"""
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
        x_predicted = thevenin_model(1, x)
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

""" 

discharged = False
i = 0
while not discharged:
    delta_t = time[i+1] - time[i]  # Time step
    I = current[i]  # Current at time i
    V = voltage[i]  # Voltage at time i

    state_model = state_transition(x, I, V, delta_t, capacity, R0, RC)
    V_thevenin_predicted = state_model[2, 0]
    R_thevenin_predicted = state_model[3, 0]
    V_predicted = V_thevenin_predicted - I * R_thevenin_predicted

    EK = ExtendedKalmanFilter(dim=, dim=)
    ekfmodel = 
    for i in range(1, len(battery_data)):
        ekfmodel.update( , , )


        # Save the estimated SoC
        SoC.append(x[0, 0])

        # Check if the battery has discharged
        if SoC[-1] <= 0:
            discharged = True
        else:
            # Update the state variables for the next iteration
            OCV.append(x[1, 0])
            V_thevenin = x[2, 0]
            R_thevenin = x[3, 0]
            i += 1

    






