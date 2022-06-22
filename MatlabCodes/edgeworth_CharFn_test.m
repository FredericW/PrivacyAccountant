% ----------------------------------------------
% edgeworth_CharFn_test.m
% last updated: 11/01/2021
% ----------------------------------------------

% This is a edgeworth expansion test function, all parameters are verified
% to be valid.

% ----------------------------------------------

function y = edgeworth_CharFn_test(a,t,lambda)
P_1 =  1/sqrt(n)^1 * (1/6*lambda(3).*(a+1i*t).^3);
P_2 =  1/sqrt(n)^2 * (1/24 *lambda(4).*(a+1i*t).^4 ...
    + 1/72 *lambda(3)^2.*(a+1i*t).^6);
P_3 =  1/sqrt(n)^3 * (1/120 *lambda(5).*(a+1i*t).^5 ...
    + 1/144 *lambda(3)*lambda(4).*(a+1i*t).^7 ...
    + 1/1296 *lambda(3)^3.*(a+1i*t).^9);
P_4 =  1/sqrt(n)^4 * (1/720 *lambda(6).*(a+1i*t).^6 ...
    + (1/1152 * lambda(4)^2 + 1/720 * lambda(3) *lambda(5)).*(a+1i*t).^8 ...
    + 1/1728 *lambda(3)^2 * lambda(4).*(a+1i*t).^10 ...
    + 1/31104 * lambda(3)^4.* (a+1i*t).^12);
y =  (1+ P_1+ P_2+P_3+P_4).*exp(1/2*(a+1i*t).^2);
end
