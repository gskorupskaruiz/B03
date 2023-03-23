function [fitresult, gof] = createFit(SOC, OCV)
%CREATEFIT(SOC,OCV)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input: SOC from SOC_OCV
%      Y Output: OCV from SOC_OCV
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  See also FIT, CFIT, SFIT.

%  Auto-generated by MATLAB on 23-Mar-2023 09:20:59


%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( SOC, OCV );

% Set up fittype and options.
ft = fittype( 'fourier7' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3.49065850398866];

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
%figure( 'Name', 'untitled fit 1' );
%h = plot( fitresult, xData, yData );
%legend( h, 'OCV vs. SOC', 'untitled fit 1', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
%xlabel( 'SOC', 'Interpreter', 'none' );
%ylabel( 'OCV', 'Interpreter', 'none' );
%grid on


