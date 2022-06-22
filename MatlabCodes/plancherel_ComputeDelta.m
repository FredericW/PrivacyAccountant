% ----------------------------------------------
% edgeworth_CharFn.m
% last updated: 11/01/2021
% ----------------------------------------------

% This function give the edgeworth approximation of the c.f. given constant
% a, interested points s, and lambda.

% ----------------------------------------------   

function y=plancherel_ComputeDelta(a,eps,cf,tmax)

filter = @(t) exp(-(a+1i*t).*eps)./(a+1i*t)./(1+a+1i*t);

w = chebfun(@(t) cf(t).*filter(t),[-tmax,tmax],'splitting','on');

y = (sum(w)/pi/2);

end