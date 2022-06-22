function plancherel_PlotFilter(a,cumulant,tmax,epsmax,N)
aa = linspace(0,epsmax);
bb = linspace(-tmax,tmax);
[eps,t] = meshgrid(aa,bb);
filter = sqrt(N*cumulant(2))*exp(-(a+1i*t).*(eps-N*cumulant(1))./sqrt(N*cumulant(2)))./(a+1i*t)./(sqrt(N*cumulant(2))+a+1i*t);
surf(aa,bb,real(filter))
end