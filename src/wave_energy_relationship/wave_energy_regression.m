%% Wave Energy Regression Sample Code - OCE2901
% By: Braidan Duffy 02/05/2023

close all; clear all;

% We want to perform a basic linear regression on some sample wave power
% measurements we make. For sake of example, we will assume that we are
% measuring the wave height and period as separate independent variables.

T = linspace(8,16,80); % True wave periods
H = linspace(0.1,5,80); % True wave heights
P = 1025 * 9.81^2 .* T .* H.^2 / (32*pi); % Calculated/true wave power (assuming sea water)
H_sample = H + 0.1*randn(size(H)); % Measured wave heights
T_sample = T + 0.1*randn(size(T)); % Measured wave periods
P_sample = 1025 * 9.81^2 .* T_sample .* H_sample.^2 / (32*pi); % Measured wave power (assuming noisy instruments/error)

%% Now we can plot the variables together to see their relationship
hold on
grid on
plot3(H, T, P, 'color', 'red', 'linestyle', '-.') % Plot the true wave power
plot3(H, T, P_sample, 'color', 'blue') % Plot the "measured" wave power with noise
xlabel("Wave Height [m]")
ylabel("Wave Period [s]")
zlabel("Wave Power per Meter [W]")
legend("True Power", "Measured Power")
hold off

%% We can perform a basic exponential regression
% By using the MATLAB curve fitting toolbox, we can determine the curve of
% best fit for our data. Since the data is exponential, we have to define a
% custom equation: z = a * x^2 * y. 

% The curve gives us a value for 'a' of 977.5. The correct value should be
% 981 - a close match! We also get an R^2 value of 0.9953 and an RMSE of
% 7995 (2%), which all indicate that the sample data closely follows the
% expected trend!
% for 'a' of 
