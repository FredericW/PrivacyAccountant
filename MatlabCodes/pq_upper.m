function [p2,q2]=pq_upper(Eg,gamma_grid2)

Eg_grid2 = arrayfun(Eg,gamma_grid2);

% We are ready to compute the new (p3,q3) pairs, which is unifom on log(gamma)
aa =(Eg_grid2(3:end)-Eg_grid2(2:end-1))./(gamma_grid2(3:end)-gamma_grid2(2:end-1));
bb =(Eg_grid2(2:end-1)-Eg_grid2(1:end-2))./(gamma_grid2(2:end-1)-gamma_grid2(1:end-2));
% this was a little wrong. Here I'm fixing it
init_slope = (Eg_grid2(1)-1)/gamma_grid2(1);
q2 = (aa-bb);
q2 = [1+init_slope,bb(1)-init_slope,q2,-aa(end)];
p2=[0,q2(2:end).*gamma_grid2];
q2=[q2,0];
p2=[p2,max(0,1-sum(p2))];

end
