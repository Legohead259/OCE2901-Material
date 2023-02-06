% Calculates the particle trajectories, velocities, and pressures beneath a
% given design wave at a specific depth or in 10 partitions from z=0 to z=-h.

% @param H:     the design wave height [m]
% @param T:     the design wave period [s]
% @param h:     the design water depth [m]
% @param z:     the desired water depth for information [m] (Fedault: 0)
% @param g:     gravitation acceleration [m/s/s] (Default: 9.81)
% @param rho:   fluid density [kg/m^3] (Default: 1025)
%
% @return Sx:   Maximum particle x-displacements [m]
% @return Sz:   Maximum particle z-displacements [m]
% @return u:    Maximum horizontal velocities [m/s]
% @return w:    Maximum vertical velocities [m/s]
% @return ax:   Maximum horizontal accelerations [m/s/s]
% @return ax:   Maximum vertical accelerations [m/s/s]
% @return Kp:   Pressure reduction factor underneath wave at depth
% @return Pd:   Maximum dynamic pressure underneath wave [Pa]
% @return Ph:   Maximum hydrostatic pressure underneath wave [Pa]
% @return P:    Pressure underneath wave [Pa]
function [Sx, Sz, u, w, ax, az, Kp, Pd, Ph, P] = wave_particles(H, T, h, z, g, rho)
    arguments
        % TODO: Argument validation
        H
        T
        h
        z = 0
        g = 9.81; % m/s^2
        rho = 1025; % kg/m^3
    end
    
    [~,~,~,k] = dispersion(T, h, g);                % Calculate wave number with dispersion equation
    sigma = 2*pi/T;                                 % Calculate angular frequency
    Sx = -H/2 * cosh(k*(h+z))/sinh(k*h);            % Calculate max x-displacement
    Sz = H/2 * sinh(k*(h+z))/sinh(k*h);             % Calculate max z-displacement
    u = H/2 * g*k/sigma * cosh(k*(h+z))/cosh(k*h);  % Calculate maximum x velocity
    w = H/2 * g*k/sigma * sinh(k*(h+z))/cosh(k*h);  % Calculate maximum z velocity
    ax = H/2 * g*k * cosh(k*(h+z))/cosh(k*h);       % Calculate maxium x acceleration
    az = -H/2 * g*k * sinh(k*(h+z))/cosh(k*h);      % Calculate maxium z acceleration
    Kp = cosh(k*(h+z))/cosh(k*h);                   % Calculate pressure response factor
    Pd = rho*g*H/2*Kp;                              % Calculate maximum hydrodynamic pressure
    Ph = -rho*g*z;                                  % Calculate maximum hydrostatic pressure
    P = Pd + Ph;                                    % Calculate total maximum pressure
end