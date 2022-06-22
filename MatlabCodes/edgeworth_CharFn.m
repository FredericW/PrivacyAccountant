% ----------------------------------------------
% edgeworth_CharFn.m
% last updated: 11/01/2021
% ----------------------------------------------

% This function give the edgeworth approximation of the c.f. given constant
% a, interested points s, and lambda.

% ----------------------------------------------

function y=edgeworth_CharFn(a,t,s,lambda,N)

p_j=0;
count_P=zeros(s-2);
div = zeros(s-2);
sum_P = 0;
Psum=0;
for k = 1:s-3
    div = zeros(s-3);
    dividing_recursion(1);
    Psum = Psum + generate_P(k)./ (sqrt(N).^k);
end

y=exp(0.5*(a+1i*t).^2).*(1+Psum);

% This function generates the P_k terms

    function y=generate_P(k)
        sum_P2=0;
        for num=1:count_P(k)
            prod=1;
            for j=1:k
                if p_j(k,num,j)~=0
                    prod=prod * 1/factorial(p_j(k,num,j)) ...
                        *(lambda(j+2)./factorial(j+2)).^(p_j(k,num,j))...
                        .*(1i*t).^(p_j(k,num,j)*(j+2));
                end
            end
            sum_P2=sum_P2+prod;
        end
        y=sum_P2;
    end


    function dividing_recursion(j)
        if sum_P==k
            count_P(k)=count_P(k)+1;
            for i1=1:k
                p_j(k,count_P(k),i1)=length(find(div == i1*ones(size(div))));
            end
            return
        end
        for i2=1:k
            if (sum_P+i2) <= k && div(j) <= i2
                div(j+1)=i2;
                sum_P=sum_P+div(j+1);
                dividing_recursion(j+1);
                sum_P=sum_P-div(j+1);
                div(j+1)=0;
            elseif div(j)>i2
                continue
            else
                return
            end
        end
    end
end