function y = ma_ComputeEps(orders,rdp,delta)

if delta <=0
    printf("Warning: Delta must be greater than zero!")
end

if length(orders)~=length(rdp)
    printf("Warning: Inputs must have the same size!")
end

eps_vec = zeros(1,length(rdp));

for i=1:length(rdp)
    
    if orders(i)<1
        printf("Warning: Renyi divergence order must be greater than 1!")
    end
    
    if rdp(i)<0
        printf("Warning: Renyi divergence must be greater than 0!")
    end
    
    
    if delta^2+exp(-rdp(i))-1>=0
        eps=0;
    elseif orders(i)>1.01
        eps= rdp(i)+log(1-1/orders(i))-log(delta*orders(i))/(orders(i)-1);
    else
        eps=inf;
    end
    eps_vec(i)=eps;
end

y=max(min(eps_vec),0);

end