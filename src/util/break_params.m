% Determines the breaking characteristics of the wave (breaking water depth
% and breaking wave height) coming from offshore onto shore.
%
% @param H0p:   The unrefracted deep-water wave height [m]
% @param H0:    The deep-water wave height [m]
% @param L0:    The deep-water wavelength [m]
% @param T:     The wave period [s]
% @param beta:  The bottom slope [°]
% @param g:     Gravitational acceleration [m/s/s] (Default: 9.81)
%
% @return Hb: the breaking wave height [m]
% @return hb: the breaking water depth [m]
function [Hb, hb] = break_params(H0p, H0, L0, T, beta, g)
    arguments
        H0p
        H0
        L0
        T
        beta
        g = 9.81; % m/s^2
    end
    
    omega_b = 0.56 * (H0p/L0)^(-1/5);       % Calculate breaker index
    Hb = omega_b * H0;                      % Calculate breaking wave height
    a = 43.8 * (1-exp(-19*tand(beta)));     % Calculate a-coefficient
    b = 1.56 / (1+exp(-19.5*tand(beta)));   % Calculate b-coefficient
    gamma_b = b - a * (Hb/(g*T^2));         % Calculate breaker depth index
    hb = Hb / gamma_b;                      % Calculate breaking water depth
end