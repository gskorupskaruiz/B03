import numpy as np
import pandas as pd
import scipy as s
from scipy.interpolate import griddata


def EKF_algorithm(Current, Vt_Actual, Temperature):

    bm = pd.read_excel("C:/Users/gowri/OneDrive - Delft University of Technology/BSC-2/AE2224/Prognostics/battery_model.xlsx") #Load the battery parameters 
    sococv = pd.read_excel("C:/Users/gowri/OneDrive - Delft University of Technology/BSC-2/AE2224/Prognostics/SOC-OCV.xlsx") #Load the SOC-OCV curve

    SOC_Init    = 1 #intial SOC
    X           = [SOC_Init, 0, 0] #state space x parameter intializations
    DeltaT      = 1 #sample time in seconds
    Qn_rated    = 2.3 * 3600 #Ah to Amp-seconds

    
    # initialize scatteredInterpolant functions for battery parameters and SOC-OCV curve
    # this function also allows for extrapolation
    F_R0    = griddata(bm['t'], bm['SOC'], bm['R0'])
    F_R1    = griddata(bm['t'], bm['SOC'], bm['R1'])
    F_R2    = griddata(bm['t'], bm['SOC'], bm['R2'])
    F_C1    = griddata(bm['t'], bm['SOC'], bm['C1'])
    F_C2    = griddata(bm['t'], bm['SOC'], bm['C2'])
    # F_OCV   = scatteredInterpolant(param.T,param.SOC,param.OCV)  
    # OCV can be extrapolated using the same method or through the polyfit function

    SOCOCV  = np.polyfit(sococv['SOC'], sococv['OCV'], 11) # calculate 11th order polynomial for the SOC-OCV curve 
    dSOCOCV = np.polyder(SOCOCV) # derivative of SOC-OCV curve for matrix C

    n_x   = np.size(X, 1)
    R_x   = 2.5e-5
    P_x   = np.array([[0.025, 0, 0],
                      [0, 0.01, 0],
                      [0, 0, 0.01]])
    Q_x   = np.array([[1.0, e-6, 0],
                      [0, 1.0e-5, 0], 
                      [0, 0, 1.0e-5]])

    SOC_Estimated   = []
    Vt_Estimated    = []
    Vt_Error        = []
    ik              = len(Current)
    #Current         = Current-0.1;

    for k in range(1, ik+1):
        T           = Temperature(k) # C
        U           = Current(k) # A
        SOC         = X(1)
        V1          = X(2)
        V2          = X(3)
    
        # Evaluate the battery parameter scatteredInterpolant 
        # functions for the current temperature & SOC
        R0     = F_R0(T,SOC)
        R1     = F_R1(T,SOC)
        R2     = F_R2(T,SOC)
        C1     = F_C1(T,SOC)
        C2     = F_C2(T,SOC)
        # OCV    = F_OCV(T,SOC)
        # OCV    = pchip(param.SOC,param.OCV,SOC) % pchip sample for unknown or single temperature
    
        OCV = np.polyval(SOCOCV,SOC) # calculate the values of OCV at the given SOC, using the polynomial SOCOCV

        Tau_1       = C1 * R1
        Tau_2       = C2 * R2
    
        a1 = np.exp(-DeltaT/Tau_1)
        a2 = np.exp(-DeltaT/Tau_2)
    
        b1 = R1 * (1 - np.exp(-DeltaT/Tau_1))
        b2 = R2 * (1 - np.exp(-DeltaT/Tau_2)) 

        TerminalVoltage = OCV - R0*U - V1 - V2

        if U > 0:
            eta = 1 # eta for discharging
        elif U <= 0:
            eta = 1 # eta for charging
        

        dOCV = np.polyval(dSOCOCV, SOC)
        C_x    = np.array([dOCV -1 -1])

        Error_x   = Vt_Actual(k) - TerminalVoltage

        Vt_Estimated    = [Vt_Estimated, TerminalVoltage]
        SOC_Estimated   = [SOC_Estimated, X(1)]
        Vt_Error        = [Vt_Error, Error_x]

        A   = np.array([[1, 0,  0],
                        [0, a1, 0],
                        [0, 0,  a2]])
    
        B   = [-(eta * DeltaT/Qn_rated), b1, b2]
        X   = (A * X) + (B * U)
        P_x = (A * P_x * A.transpose()) + Q_x

        KalmanGain_x = (P_x) * (C_x.transpose()) * np.linalg.inv(((C_x * P_x * C_x.transpose()) + (R_x)))
        X            = X + (KalmanGain_x * Error_x)
        P_x          = (np.eye(n_x,n_x) - (KalmanGain_x * C_x)) * P_x


    return SOC_Estimated, Vt_Estimated, Vt_Error
     





