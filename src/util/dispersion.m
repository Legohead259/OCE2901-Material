% Calculates the theoretical wavelength using the dispersion equation:
% L = L0 * tanh( 2*pi*h / L )

% @param T:     the wave period [s]
% @param h:     the water depth [m]
% @param g:     the graviational acceleration [m/s/s] (Default: 9.81)
% @param tol:   the accuracy tolerance of the calculation [m] (Default: 0.001)

% @return L_0:      the deepwater wavelength [m]
% @return L:        the design wavelength [m]
% @return k0:       the deep water wave number
% @return k:        the design wavenumber
% @return kh0:      the deep water relative depth
% @return kh:       the design relative depth
% @return num_iter: the number of iterations it took to calculate the wavelength
function [L0, L, k0, k, kh0, kh, num_iter] = dispersion(T, h, g, tol)
    arguments
        T
        h
        g = 9.81
        tol = 0.001 % ±1 mm
    end
    
    num_iter = 0;
    L0 = g * T^2 / (2 * pi); % Define the deepwater wavelength
    k0 = 2*pi/L0;
    kh0 = k0*L0/2;
    y = 0; % Instantiate an interim temporary value
    L = L0; % Start the iteration with the deep water wavelength
    
    while(abs(L-y) > tol)           % Iterate through dispersion equation until tolerance is reached
        y = L;                      % Save previous iteration of L
        L = L0 * tanh(2*pi*h / L);  % Define next iteration of L
        L = abs((L+y) / 2);         % Find average between calculations to get towards tolerance
        num_iter = num_iter + 1;    % Increment number of iterations
    end                             % Return the final L value as the theoretical wavelength
    k = 2*pi/L;                     % Calculate wave number
    kh = k*h;                       % Calculate relative depth
end

