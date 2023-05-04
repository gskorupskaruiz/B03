
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy as s
from scipy import interpolate
from scipy.linalg import svd

K = 100
def EKF_algorithm():
    bm = pd.read_excel("Mark/battery_model.xlsx")  # Load the battery parameters
    sococv = pd.read_excel("Mark/SOC_OCV_data.xlsx")  # Load the SOC-OCV curve

    SOC_Init = 1  # intial SOC
    X = np.array([1, 0, 0])  # state space x parameter intializations
    DeltaT = 1  # sample time in seconds
    Qn_rated = 2.3 * 3600  # Ah to Amp-seconds

    # initialize scatteredInterpolant functions for battery parameters and SOC-OCV curve
    # this function also allows for extrapolation
    '''
    F_R0 = interpolate.Rec(bm['T'].to_numpy(), bm['SOC'].to_numpy(), bm['R0'].to_numpy())
    F_R1 = interpolate.interp2d(bm['T'].to_numpy(), bm['SOC'].to_numpy(), bm['R1'].to_numpy())
    F_R2 = interpolate.interp2d(bm['T'].to_numpy(), bm['SOC'].to_numpy(), bm['R2'].to_numpy())
    F_C1 = interpolate.interp2d(bm['T'].to_numpy(), bm['SOC'].to_numpy(), bm['C1'].to_numpy())
    F_C2 = interpolate.interp2d(bm['T'].to_numpy(), bm['SOC'].to_numpy(), bm['C2'].to_numpy())
    '''
    # F_OCV   = scatteredInterpolant(param.T,param.SOC,param.OCV)
    # OCV can be extrapolated using the same method or through the polyfit function
    F_R0 = interpolate.interp2d(bm['T'].to_numpy(), bm['SOC'].to_numpy(), bm['R0'].to_numpy(), kind='cubic')
    F_R1 = interpolate.interp2d(bm['T'].to_numpy(), bm['SOC'].to_numpy(), bm['R1'].to_numpy(), kind='cubic')
    F_R2 = interpolate.interp2d(bm['T'].to_numpy(), bm['SOC'].to_numpy(), bm['R2'].to_numpy(), kind='cubic')
    F_C1 = interpolate.interp2d(bm['T'].to_numpy(), bm['SOC'].to_numpy(), bm['C1'].to_numpy(), kind='cubic')
    F_C2 = interpolate.interp2d(bm['T'].to_numpy(), bm['SOC'].to_numpy(), bm['C2'].to_numpy(), kind='cubic')

    SOCOCV = np.polyfit(sococv['SOC'].to_numpy(), sococv['OCV'].to_numpy(), 6)  # calculate 11th order polynomial for the SOC-OCV curve4'
    
    dSOCOCV = np.polyder(SOCOCV)  # derivative of SOC-OCV curve for matrix C

    n_x = np.shape(X)[0]
    R_x = 2.5e-5
    P_x = np.array([[0.025, 0, 0],
                    [0, 0.01, 0],
                    [0, 0, 0.01]])
    Q_x = np.array([[1.0, 1e-6, 0],
                    [0, 1.0e-5, 0],
                    [0, 0, 1.0e-5]])

    SOC_Estimated = []
    Vt_Estimated = []
    Vt_Error = []
    # ik              = len(Current)
    # Current         = Current-0.1;

    df = pd.read_csv(r'data/B0005_TTD.csv')
    Vt_Actual = df['Voltage_measured'].to_numpy()[:K]
    Measured_Current = df['Current_measured'].to_numpy()[:K]
    Temperature = df['Temperature_measured'].to_numpy()[:K]

    # Current Definition: (+) Discharging, (-) Charging
    Current = - Measured_Current

    ik = len(Current)

    for k in range(ik):
        T = Temperature[k]
        U = Current[k]  # A
        SOC = X[0]
        V1 = X[1]
        V2 = X[2]

        # Evaluate the battery parameter scatteredInterpolant
        # functions for the current temperature & SOC
        R0 = F_R0(T, SOC)

        R1 = F_R1(T, SOC)
        R2 = F_R2(T, SOC)
        C1 = F_C1(T, SOC)
        C2 = F_C2(T, SOC)
        # OCV    = F_OCV(T,SOC)
        # OCV    = pchip(param.SOC,param.OCV,SOC) % pchip sample for unknown or single temperature

        OCV = np.polyval(SOCOCV, SOC)  # calculate the values of OCV at the given SOC, using the polynomial SOCOCV

        Tau_1 = C1 * R1
        Tau_2 = C2 * R2

        a1 = np.exp(-DeltaT / Tau_1)
        a2 = np.exp(-DeltaT / Tau_2)

        b1 = R1 * (1 - np.exp(-DeltaT / Tau_1))
        b2 = R2 * (1 - np.exp(-DeltaT / Tau_2))

        TerminalVoltage = OCV - R0 * U - V1 - V2

        if U > 0:
            eta = 1  # eta for discharging
        elif U <= 0:
            eta = 1  # eta for charging

        dOCV = np.polyval(dSOCOCV, SOC)
        C_x = np.array([dOCV, -1, -1])

        Error_x = Vt_Actual[k] - TerminalVoltage

        Vt_Estimated.append(TerminalVoltage)
        SOC_Estimated.append(X[0])
        #Vt_Error = [Vt_Error, Error_x]

        A = np.matrix([[1, 0, 0],
                       [0, a1[0], 0],
                       [0, 0, a2[0]]
                       ])

        B = [-(eta * DeltaT / Qn_rated), b1[0], b2[0]]
        # print(np.matrix(X).reshape(3,1))
        X_alt = (A @ np.array(X).reshape(3,1)) + (np.array(B).reshape(3,1) * U)

        P_x = (A @ P_x @ A.T) + Q_x
        #print(P_x.shape)

        KalmanGain_x = P_x @ C_x.reshape(3, 1) @ np.linalg.inv(C_x.reshape(1,3) @ P_x @ C_x.reshape(3,1) + R_x)

        #P_x = (A @ P_x @ np.matrix(A).H) + Q_x
        #KalmanGain_x = P_x @ np.matrix(C_x.reshape(1,3)).H @ np.linalg.inv(C_x @ P_x @ np.matrix(C_x.reshape(1,3)).H + R_x)
        #z = (((np.matrix(C_x) @ P_x @ np.matrix(C_x).reshape(3, 1)) + (R_x) * np.matrix(np.ones((3, 3)))))
        #KalmanGain_x = (P_x) @ z @ np.matrix(C_x).reshape(3,1)
        X_alt = X_alt + (KalmanGain_x * Error_x)
        P_x = (np.eye(3, 3) - KalmanGain_x @ np.array(C_x).reshape(1,3)) @ P_x

    return SOC_Estimated, Vt_Estimated, Vt_Error

df = pd.read_csv(r'data/B0005_TTD.csv')
RecordingTime = df['Time'].to_numpy()[:K]
Measured_Voltage = df['Voltage_measured'].to_numpy()[:K]
Measured_Current = df['Current_measured'].to_numpy()[:K]
Measured_Temperature = df['Temperature_measured'].to_numpy()[:K]

nominalCap = 2.3  # Battery capacity in Ah taken from data.

# Current Definition: (+) Discharging, (-) Charging
Measured_Current_R = - Measured_Current

# Converting seconds to hours
RecordingTime_Hours = RecordingTime / 3600

SOC_Estimated, Vt_Estimated, Vt_Error = EKF_algorithm()

# Terminal Voltage Measured vs. Estimated

plt.plot(RecordingTime_Hours, Measured_Voltage, label='measured')
plt.plot(RecordingTime_Hours, Vt_Estimated, label='est')
plt.legend()
plt.ylabel('Terminal Voltage[V]')
plt.xlabel('Time[hr]')
plt.title('Measured vs. Estimated Terminal Voltage (V)')
plt.show()

"""
% Terminal Voltage Error
figure
% plot(LiPoly.RecordingTime_Hours,Vt_Error);
% legend('Terminal Voltage Error');
ylabel('Terminal Voltage Error');
xlabel('Time[hr]');
% SOC Coulomb Counting vs. Estimated
% figure
% plot (LiPoly.RecordingTime_Hours,LiPoly.Measured_SOC);
hold on
plot (LiPoly.RecordingTime_Hours,SOC_Estimated*100);
hold off;
% legend('Coulomb Counting','Estimated EKF');
ylabel('SOC[%]');xlabel('Time[hr]');
title('SOC Estimated ')
grid minor
% Vt Error
figure
plot(LiPoly.RecordingTime_Hours, Vt_Error);
legend('Vt Error');
ylabel('vt Error [%]');
xlabel('Time[hr]');
grid minor"""