% ----------------------------------------------
% edgeworth_CharFn.m
% last updated: 11/01/2021
% ----------------------------------------------

% This function give the edgeworth approximation of the c.f. given constant
% a, interested points s, and lambda.

% ----------------------------------------------   

function y=edgeworth_ComputeDelta_cum(a,eps,cf,tmax,cumulant,N)

filter = @(t) sqrt(N*cumulant(2))*...
    exp(-(a+1i*t).*(eps-N*cumulant(1))./sqrt(N*cumulant(2)))...
    ./(a+1i*t)./(sqrt(N*cumulant(2))+a+1i*t);

w = chebfun(@(t) cf(t).*filter(t),[0,tmax],'splitting','on');

y = (sum(w)/pi);

end