function coeff=least_square_polyfit_xyz(x,y,z,m)
% By Thomas Bifano, Boston University, May 2009 tgb@bu.edu
% This function calculates the least-squares best fit polynomial as a
% function of x and y for model function of k terms:
%
% z=coeff(1) * x ^ m (1,1) * y ^ m(1,2)  + coeff(2) * x ^ m (2,1) * y ^ m(2,2) + .... + coeff(k) * x ^ m (k,1) * y ^ m(k,2)
%
% Example: A second order fit in x and y would have 6 terms (k=6): z=a+bx+cx^2+dxy+ey+fy^2
% Input includes three equal-length vectors (with more than k components) comprised of x, y, and z measured data 
% 
% The matrix m contains k pairs of exponents for the x (col 1) and y (col
% 2) exponents. The full second order model described above would have m=[0 0 ; 1 0 ; 2 0 ; 1 1 ; 0 1 ; 0 2]
%
% Second example: ifor the model equation z=ax^3 + bxy^2 +c,  k=3 (three terms) and m= [ 3 0 ; 1 2 ; 0 0 ]
% 
% No error checking. Make sure x,y,z are column vectors of equal length.

k=length(m); % number of terms
for s=1:k
    LHS(s)=sum(z.*x.^m(s,1).*y.^m(s,2));
    for r=1:k
        RHS(s,r)=sum(x.^m(r,1).*y.^m(r,2).*x.^m(s,1).*y.^m(s,2));
    end
end
coeff=(LHS/RHS)';
        
        