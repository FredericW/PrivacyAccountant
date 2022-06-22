% ----------------------------------------------
% cumulants_generatorDiscrete.m
% last updated: 05/23/2022
% ----------------------------------------------

% this function takes as input the discrete pdf and logloss,
% and the orders of moments/cumulants etc we interested in.

% ----------------------------------------------

function [moment, cumulant, lambda, rho]=cumulants_generatorDiscrete(ps,logloss,orders)


moment=zeros(1,orders);
moment_abs = zeros(1,orders);

for i=1:orders
    moment(i) = sum(ps.*logloss.^i);
    moment_abs(i)=sum(ps.*abs(logloss).^i);
end

cumulant = zeros(1,orders);
cumulant(1)=moment(1);
for i=2:orders
    buf=0;
    for m=1:1:i-1
        buf=buf-nchoosek(i-1,m-1).*cumulant(m).* moment(i-m);
    end
    cumulant(i)=moment(i)+buf;
end

rho = zeros(1,orders);
lambda = zeros(1,orders);
for i=1:orders
    rho(i)=moment_abs(i)./(sqrt(cumulant(2)).^i);
    lambda(i)=cumulant(i)./(sqrt(cumulant(2)).^i);
end
end
