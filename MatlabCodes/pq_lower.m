function [p3,q3]=pq_lower(Eg,Egslope,gamma_grid2)

f_grid = zeros(size(gamma_grid2));
f_grid(1) = Eg(gamma_grid2(1));
f_grid(end) = 0;

for i=2:length(f_grid)-1
%     slp = (f_grid(end)-f_grid(i-1))/(gamma_grid2(end)-gamma_grid2(i-1));
    myfnc = @(gamma)(Eg(gamma)-f_grid(i-1))./(gamma-gamma_grid2(i-1))-Egslope(gamma);

    tmin = gamma_grid2(i-1)+eps(gamma_grid2(i-1));
    tmax = gamma_grid2(i);
    % should be that myfnc(tmin) > 0 and myfnc(tmax) < 0

    if myfnc(tmax) >= 0 % for most of the case this should be true
        f_grid(i) = Eg(tmax);
    else
        if myfnc(tmin) <= 0
            slp = Egslope(gamma_grid2(i-1));
        else
            t = fzero(myfnc,[tmin,tmax]);
            slp = (Eg(t)-f_grid(i-1))./(t-gamma_grid2(i-1));
        end
        f_grid(i) = max(0,f_grid(i-1)+slp*(gamma_grid2(i)-gamma_grid2(i-1)));
    end

end

% The above produces a function that is not necessarily convex. Here we convexify it
gamma_grid_ext = [0,gamma_grid2];
f_grid_ext = [1,f_grid];
K = convhull(gamma_grid_ext,f_grid_ext);

slope = (f_grid_ext(K(2:end))-f_grid_ext(K(1:end-1)))...
    ./(gamma_grid_ext(K(2:end))-gamma_grid_ext(K(1:end-1)));
q3 = zeros(1,length(gamma_grid2));
q3(K(2:end-2)-1) = slope(2:end-1)-slope(1:end-2);
q3(end) = -slope(end-1);

p3 = q3.*(gamma_grid2(1:end));
end
