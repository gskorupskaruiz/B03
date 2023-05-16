clc; clear; close all;

load('B0030.mat');
t = readtable('B0005_TTD.csv');
fieldnames(t);

LiPoly.RecordingTime            = B0030.cycle(2).data.Time;
LiPoly.Measured_Voltage         = B0030.cycle(2).data.Voltage_measured;
LiPoly.Measured_Current         = B0030.cycle(2).data.Current_measured;
LiPoly.Measured_Temperature     = B0030.cycle(2).data.Temperature_measured;


% LiPoly.RecordingTime_alt            = table2array(readtable('B0005_TTD.csv', 'range', 'F1:F198')); %readtable('B0005_TTD.csv', 'range', 'F1:F198');
% LiPoly.Measured_Voltage_alt         = table2array(readtable('B0005_TTD.csv', 'range', 'A1:A198')); %t.('Voltage_measured');
% LiPoly.Measured_Current_alt         = table2array(readtable('B0005_TTD.csv', 'range', 'B1:B198')); %t.('Current_measured');
% LiPoly.Measured_Temperature_alt     = table2array(readtable('B0005_TTD.csv', 'range', 'C1:C198')); %t.('Temperature_measured');


nominalCap                      = 2.3; % Battery capacity in Ah taken from data.

% Current Definition: (+) Discharging, (-) Charging
LiPoly.Measured_Current_R       = - LiPoly.Measured_Current;
% LiPoly.Measured_Current_R_alt       = -LiPoly.Measured_Current_alt;

% Converting seconds to hours
LiPoly.RecordingTime_Hours      = LiPoly.RecordingTime/3600;

% LiPoly.RecordingTime_Hours_alt     = LiPoly.RecordingTime_alt/3600;

[SOC_Estimated, Vt_Estimated, Vt_Error] = EKFalgorithm(LiPoly.Measured_Current_R, LiPoly.Measured_Voltage, LiPoly.Measured_Temperature);
[SOC_Estimated_1, Vt_Estimated_1, Vt_Error_1] = EKFalgorithm(-2, 1.89, 24+274);
% [SOC_Estimated_alt, Vt_Estimated_alt, Vt_Error_alt] = EKFalgorithm(LiPoly.Measured_Current_R_alt, LiPoly.Measured_Voltage_alt, LiPoly.Measured_Temperature_alt);

disp(SOC_Estimated_1)
disp(Vt_Estimated_1)
disp('a')

%[SOC_Estimated_alt, Vt_Estimated_alt, Vt_Error_alt] = EKFalgorithm(LiPoly.Measured_Current_R_alt, LiPoly.Measured_Voltage_alt, LiPoly.Measured_Temperature_alt);

% Terminal Voltage Measured vs. Estimated
figure
%plot(LiPoly.RecordingTime_Hours,LiPoly.Measured_Voltage);
%hold on;
plot(LiPoly.RecordingTime_Hours,LiPoly.Measured_Voltage);
hold on;
plot(LiPoly.RecordingTime_Hours,Vt_Estimated);
%hold on
%plot(LiPoly.RecordingTime_Hours,Vt_Estimated);
hold off;
legend('Measured', 'Estimated EKF');
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
plot (LiPoly.RecordingTime_Hours, SOC_Estimated*100);
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

voltage_B0029 = [];
soc_B0029 = [];

for i=1:1:97
    load('B0029.mat');

    % if B0007.cycle(i).type== 'discharge';
    if strcmp(B0029.cycle(i).type, 'discharge')

        LiPoly.RecordingTime            = B0029.cycle(i).data.Time;
        LiPoly.Measured_Voltage         = B0029.cycle(i).data.Voltage_measured;
        LiPoly.Measured_Current         = B0029.cycle(i).data.Current_measured;
        LiPoly.Measured_Temperature     = B0029.cycle(i).data.Temperature_measured;

        LiPoly.Measured_Current_R       = - LiPoly.Measured_Current;

        LiPoly.RecordingTime_Hours      = LiPoly.RecordingTime/3600;

        [SOC_Estimated, Vt_Estimated, Vt_Error] = EKFalgorithm(LiPoly.Measured_Current_R, LiPoly.Measured_Voltage, LiPoly.Measured_Temperature);
    
        soc_B0029   = [soc_B0029;SOC_Estimated];
        voltage_B0029 = [voltage_B0029;Vt_Estimated];

    end 
%     RMSE_Vt = sqrt((sum((LiPoly.Measured_Voltage - Vt_Estimated).^2)) /(length(LiPoly.Measured_Voltage)))*1000; % mV
% 
%     Max_Vt = max(abs(LiPoly.Measured_Voltage - Vt_Estimated))*1000; % mV
%     x = [x; RMSE_Vt];

end





    

