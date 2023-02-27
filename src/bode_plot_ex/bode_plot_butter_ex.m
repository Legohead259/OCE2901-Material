%% Bode Plot Example - OCE2901
% By: Braidan Duffy
% Date Created: 2023-02-07
% Last Revision: 2023-02-07

clear all; close all; clf;

fc = 50; % Cut-off frequency
fs = 6283; % Sample frequency

[b, a] = butter(2, fc/(fs/2)); % Butterworth of order 6

freqz(b, a, [], fs); % Bode Plot
subplot(2,1,1)
ylim([-100, 20])

x = 0:0.001:2*pi;
data_in = sin(10*x) + randn(size(x))*0.1;
data_out = filtfilt(b, a, data_in);

figure
hold on
plot(x, data_in)
plot(x, data_out)
plot(x, sin(10*x), 'color', 'k', 'linestyle', '-.')
legend("raw", "filtered", "true")
xlabel("Time [s]")
ylabel("Surface Elevation [m]")
hold off