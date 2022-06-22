% ----------------------------------------------
% cumulants_generatorDiscrete.m
% last updated: 05/23/2022
% ----------------------------------------------

% this function takes as input the pdf and logloss,
% and the orders of moments/cumulants etc we interested in.

% ----------------------------------------------

function [moment, cumulant, lambda, rho]=cumulants_generator(f,q,d,s,xmax)

LogLoss = @(x) log(f(x)./(q*f(x+d)+(1-q)*f(x))); 

moment=zeros(1,s);
moment_abs = zeros(1,s);
for i=1:s
    moment(i) = sum(chebfun(@(x) f(x).*(LogLoss(x)).^i,[-xmax,xmax],'splitting','on'));
    moment_abs(i)=sum(chebfun(@(x)f(x).*abs(LogLoss(x)).^i,[-xmax,xmax],'splitting','on'));
end

cumulant = zeros(1,s);
cumulant(1)=moment(1);
for i=2:s
    buf=0;
    for m=1:1:i-1
        buf=buf-nchoosek(i-1,m-1).*cumulant(m).* moment(i-m);
    end
    cumulant(i)=moment(i)+buf;
end

rho = zeros(1,s);
lambda = zeros(1,s);
for i=1:s
    rho(i)=moment_abs(i)./(sqrt(cumulant(2)).^i);
    lambda(i)=cumulant(i)./(sqrt(cumulant(2)).^i);
end
end