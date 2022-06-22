function y=ma_ComputeDelta(orders,rdp,eps)
if eps <=0
    printf("Warning: eps must be greater than zero!")
end
if length(orders)~=length(rdp)
    printf("Warning: Inputs must have the same size!")
end
logdeltas = zeros(1,length(rdp));
for i=1:length(rdp)
    if orders(i)<1
        printf("Warning: Renyi divergence order must be greater than 1!")
    end
    if rdp(i)<0
        printf("Warning: Renyi divergence must be greater than 0!")
    end
    logdelta = 0.5*log(1-exp(-rdp(i)));
    if orders(i)>1.01
        rdp_bound = (orders(i)-1)*(rdp(i)-eps+log(1-1/orders(i)))-log(orders(i));
        logdelta=min(logdelta,rdp_bound);
    end
    logdeltas(i)=logdelta;
end
y=min(logdeltas);
y=min(exp(y),1.0);
end
