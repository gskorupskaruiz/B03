clc; clear; close all;

load('B0006.mat');
LiPoly.RecordingTime            = B0006.cycle(4).data.Time;
LiPoly.Measured_Voltage         = B0006.cycle(4).data.Voltage_measured;
LiPoly.Measured_Current         = B0006.cycle(4).data.Current_measured;
LiPoly.Measured_Temperature     = B0006.cycle(4).data.Temperature_measured;

nominalCap                      = 2.3; % Battery capacity in Ah taken from data.

% Current Definition: (+) Discharging, (-) Charging
LiPoly.Measured_Current_R       = - LiPoly.Measured_Current;

% Converting seconds to hours
LiPoly.RecordingTime_Hours      = LiPoly.RecordingTime/3600;

[SOC_Estimated, Vt_Estimated, Vt_Error] = EKFalgorithm(LiPoly.Measured_Current_R, LiPoly.Measured_Voltage, LiPoly.Measured_Temperature);

% Terminal Voltage Measured vs. Estimated
figure
plot(LiPoly.RecordingTime_Hours,LiPoly.Measured_Voltage);
hold on
plot(LiPoly.RecordingTime_Hours,Vt_Estimated);
hold off;
legend('Measured','Estimated EKF');
ylabel('Terminal Voltage[V]');xlabel('Time[hr]');
title('Measured vs. Estimated Terminal Voltage (V)')
grid minor

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
grid minor