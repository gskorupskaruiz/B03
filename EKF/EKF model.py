import numpy as np
import pandas as pd



# Step 1: Data preprocessing
df = pd.read_csv("C:/Users/gowri/OneDrive - Delft University of Technology/BSC-2/AE2224/Prognostics/B0006.csv")
time = df["Time"].values
voltage = df["Voltage_charge"][0].values
current = df["Current_charge"][0].values

# Initialize the battery parameters
capacity = 1.5  # Battery capacity (in Ah)
R0 = 0.01  # Internal resistance of the battery
RC = 30000  # Time constant of the battery's equivalent circuit
V_thevenin = 4.2  # Thevenin voltage (in V)
R_thevenin = 0.1  # Thevenin resistance (in Ohms)

# Initialize the state variables
SoC = [100]  # Start with a fully charged battery
OCV = [V_thevenin + R_thevenin * current[0]]  # Assume the battery's fully charged voltage is the Thevenin voltage
x = np.array([SoC[0], OCV[0], V_thevenin, R_thevenin]).reshape(-1, 1)

# Initialize the covariance matrix
P = np.diag([0.1, 0.1, 0.1, 0.1])

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

# Define the measurement function
def measurement(x, current, R0):
    SoC, OCV, V_thevenin, R_thevenin = x.ravel()
    V = OCV - (current * R_thevenin)
    return V

# Implement the extended Kalman filter algorithm
discharged = False
i = 790
while not discharged:
    delta_t = time[i+1] - time[i]  # Time step
    I = current[i]  # Current at time i
    V = voltage[i]  # Voltage at time i

    # Predict step
    F = np.array([[1 - (delta_t / capacity), 0, 0, 0],
                  [0, 1 - (delta_t / RC), 0, 0],
                  [0, -I / V_thevenin, 1, -I * (R_thevenin + R0) / (V_thevenin ** 2)],
                  [0, 0, 0, 1 - (delta_t / RC)]])
    x_predicted = state_transition(x, I, V, delta_t, capacity, R0, RC)
    P_predicted = np.matmul(np.matmul(F, P), F.T)

    # Update step
    V_thevenin_predicted = x_predicted[2, 0]
    R_thevenin_predicted = x_predicted[3, 0]
    V_predicted = V_thevenin_predicted - I * R_thevenin_predicted
    y = np.array([V - V_predicted, I - x_predicted[0, 0], R_thevenin - x_predicted[3, 0]])
    y = np.reshape(y, (3, 1))  # Reshape y to have shape (3, 1)
    H = np.array([[-R_thevenin_predicted, 0, 1, 0],
                  [0, -1, 0, -I * R_thevenin_predicted],
                  [-1, 0, 0, 1]])
    S = np.matmul(np.matmul(H, P_predicted), H.T) + np.eye(3) * R0
    K = np.matmul(np.matmul(P_predicted, H.T), np.linalg.inv(S))
    x = x_predicted + np.matmul(K, y)
    P = np.matmul(np.eye(4) - np.matmul(K, H), P_predicted)



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

    print(SoC)

# Print the estimated time to discharge
TTE = time[i] + (capacity * SoC[-2]) / current[i]
print("Estimated time to discharge:", TTE, "seconds")

