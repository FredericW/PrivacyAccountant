function y = ma_ComputeRDP(noise_type,orders,c_type,C,sen,q)

if noise_type== "cactus"
    filename = sprintf('cactus_x_d%.1f_%s_%.2f.csv',sen,c_type,C);
    x=csvread(filename);
    filename = sprintf('cactus_p_d%.1f_%s_%.2f.csv',sen,c_type,C);
    p=csvread(filename);
    xmax=max(x);
    n=ceil(length(x)/2/xmax);
    shift_n = ceil(n*sen);
    ps = p(shift_n:length(p)); %original distribution, padding for shift
    
    x=linspace(-xmax+sen,xmax,length(ps))';
    l1 = sum(ps.*abs(x));
    fprintf("l1 cost is %f\n",l1)
    l2 = sum(ps.*x.^2);
    fprintf("l2 cost is %f\n",l2)
    
    qs = q*p(1:length(p)-shift_n+1)+ (1-q)*ps; %shifted + subsampled dist
else
    n=500; % the quantization rate
    xmax=max(10,10*sqrt(C));
    x= linspace(-xmax,xmax,ceil(2*xmax*n))';
    switch noise_type
        case "gaussian"
            if c_type=="l2"
                sigma=sqrt(C);
            else
                sigma=C*sqrt(pi/2);
            end
            f = @(x) 1/sqrt(2*pi)/sigma*exp(-1/2*x^2/sigma^2);
        case "laplace"
            if c_type=="l2"
                b=sqrt(C/2);
            else
                b=C;
            end
            f = @(x) 1/2/b*exp(-abs(x)/b);
        case "airy"
            if c_type=="l2"
                fprintf("WARNING: You are using L2-cost!\n")
            end
            a0=1.01879;
            f = @(x)airy(2*a0/3/C*abs(x)-a0).^2/3/C/airy(-a0)^2;
    end
    ps= arrayfun(f,x)/n;
    qs= q*arrayfun(f,x-sen)/n+(1-q)*ps;
end


logmgf = @(t) log(sum(qs.*((ps./qs).^t))); %log-mgf with a fudge value of 1e-30
rdp = arrayfun(logmgf,orders);
y = (rdp)./(orders-1);

end




