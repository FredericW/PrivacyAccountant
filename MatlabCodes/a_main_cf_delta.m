% ----------------------------------------------
% main_cf_delta.m
% last updated: 12/23/2021
% ----------------------------------------------

% This file can be considered as an updated version of opt_03_main.m.

% In this file, we can 
% * define the LogLoss (privacy loss) random variable
% * compute the moment, cumulant and other relavent parameters
% * generate the characteristic function
%     * edgeworth c.f.
%     * numerical c.f.
% * compute the delta-epsilon via integrating over [-T,T]
% * generate various plots
% 
% With this code, we can set lists of noise types, sensitivities (l2_norm_limit) and stddev

% ----------------------------------------------

close all
clear

noise_list=[2,4]; % choice of noises, 1:gaussian 2:laplace 3: cactus 4:airy
sens_list=[1]; % list of sensitivities
noise_level_list=[sqrt(2)]; % the standard deviation is sensitivity* noise_level 
N = 1000; % composition
q = 0.01; % subsampling rate
s_list = [8]; % number of cumulants in approx.
noise_list_names=["Gaussian","Laplace","Cactus","Airy"];

a = 0.01; %  avoid the pole

% ----------------------------------------------

% We 
for sens_type=1:length(sens_list)
    
    for noise_level= 1:length(noise_level_list)

        d = sens_list(sens_type); % shift or sensitivity
        stddev=sens_list(sens_type)*noise_level_list(noise_level);

        variance=stddev^2; % the scale parameter for gaussian and cactus, quadric cost
        fprintf("For gaussian or cactus, the quadric cost is set to %.2f.\n",variance)
        
        b = sqrt(variance/2); % the scale parameter for laplace and airy, absolute cost
        fprintf("For laplace or airy, the absolute cost is set to %.2f.\n",b)

        for noise_type=noise_list

            fprintf('\nCurrent distribution type: %s\n',noise_list_names(noise_type))
            
            if noise_type==3
                filename = sprintf('cactus_pdf_d%.1f_v%.2f.mat',d,variance);
                load(filename,'xmax');
            else
                xmax = 10*max(1,stddev);
            end

            % note: for laplace distribution, when we consider the L1 norm, 
            % that b=sqrt(var/2) is mean of abs(X) for X~Laplace(b).

            switch noise_type
                case 1 %gaussian pdf
                    f = @(x) 1/ sqrt(2*pi)/sqrt(variance) * exp(-x.^2/2/variance);
                case 2 %laplace pdf
                    f = @(x) 1/2/b * exp(-abs(x)/b);
                otherwise %airy pdf
                    a0=1.01879;
                    C=b;
                    f = @(x)airy(2*a0/3/C*abs(x)-a0).^2/3/C/airy(-a0)^2;
            end

            % define the logloss for each pdf with given sumsampling rate
            LogLoss = @(x) log(f(x)./(q*f(x+d)+(1-q)*f(x)));                

            % compute the parameters
            [moment, cumulant, lambda, rho]=cumulants_generator(f,q,d,max(s_list),xmax);
            
            % The numerical characterstic function
            real_CharFn = @(t) sum(chebfun(@(x) f(x)...
                .*exp((a+1i.*t).*(LogLoss(x)-cumulant(1))/sqrt(N*cumulant(2))),...
                [-xmax,xmax],'splitting','on')).^N;
            
            % This is our upper bound regading the real error, details see write-up
            T_our=@(s) sqrt(N)/s/rho(s)^(1/s); % valid limits for the bound
            fprintf('T_our(3)=%.4f\n',T_our(3)); 
            
            cc = @(s) 3*exp(1)*rho(s)^(1/s)/2/s;
            s2=3; % we take s2 to be the smallest value 3
            cf_error_our =...
                @(t) s2^(s2-2)/T_our(s2).^(s2-2) .* ...
                ( (cc(s2).^(s2-2).*abs(t).^(2*(s2-2))-cc(s2).*abs(t).^(s2-1))./ (cc(s2).*abs(t)-1)...
                + cc(s2).^(s2-2).*abs(t).^(2*(s2-2))./factorial(s2-2) )...
                .* exp(-t.^2/2);
            
            % This is Kolmogorov's upper bound from his book.         
            T_kol = @(s) sqrt(N)/8/s/rho(s)^(3/s); % valid limits for the bound
            fprintf('T_kol(3)=%.4f\n', T_kol(3));
            
            cc_1= @(t) 3*(s-3)/s^2 * exp(s/8)* s^(s-2)...
                * (rho(s)^(3/s)/sqrt(N))^(s-2)...
                * (abs(t).^s + abs(t).^(3*s-6));
            cc_2 = @(t) 3^(s-2)*(rho(s)^(1/s).*abs(t)).^(3*s-6)...
                *(1/sqrt(N))^(s-2).*exp(s/8+t.^2/4);
            cf_error_kol =...
                @(t) abs(exp(-t.^2/2).*(cc_1(t)+cc_2(t)));



 
            % compute and plot the c.f.s 
            fprintf('Characteristic functions...\n')
            figure(1)

            % set the tmax for integration of eps-delta 
            % could be T_our(3) if apply our bound
            fprintf('suggestion for tmax is %.4f.\n', T_our(3))
            tmax = T_our(3); % <-- CHANGE THIS FOR DIFFERENT SETTINGS!!!            
            t_grid=linspace(0,tmax,10);

            % the numerical c.f., integration over [-xmax,xmax] by chebfun
            fprintf('numerical c.f. over [0,%.2f].\n',tmax)
            cf_real=real_CharFn(t_grid);
            semilogy(t_grid, abs(cf_real),...
                'LineWidth',1.5,'DisplayName',sprintf('%s',noise_list_names(noise_type)))
            hold on
            
%             % the edgeworth c.f. with given cumulants  
%             fprintf('edgeworth c.f. curves:\n')
%             for s=s_list
%                 fprintf('s=%d\n',s)
%                 cf_edgeworth = edgeworth_CharFn(a,t_grid,s,lambda,N);
%                 semilogy(t_grid, abs(cf_edgeworth),'--',...
%                     'LineWidth',1.5,...
%                     'DisplayName',sprintf('%s edgeworth, s=%d',noise_list_names(noise_type),s))
%                 hold on
%             end

%             % the edgeworth c.f. with our error bound, which gives an
%             % upper bound for the approximation
%             cf_edgeworth_3 = edgeworth_CharFn(a,t_grid,3,lambda,N);
%             error_our = cf_error_our(t_grid);
%             semilogy(t, min(abs(error_our+cf_edgeworth_3),1),'.',...
%                 'LineWidth',1.5,'DisplayName','Upper Bound with s=3')
%             hold on            
            



            % compute and plot the epsilon-delta curves
            fprintf('Epsilon-Delta...\n')
            figure(2)
            
            % set up the epsmax, which inspired by the formulas
            fprintf("suggestion for max eps is %.4f.\n", 8*N*cumulant(1)/sqrt(N*cumulant(2)));
            epsmax = 8*N*cumulant(1)/sqrt(N*cumulant(2)); % <-- CHANGE THIS FOR DIFFERENT SETTINGS!!!
            eps_grid = linspace(0,epsmax,20); % range of epsilon                

            % the numerical epsilon-delta, integration over [-tmax,tmax] by chebfun
            if noise_type~=3
            fprintf('plot numerical eps-delta curves over [0,%.2f].\n',epsmax)
            delta_real = delta(a,eps_grid, real_CharFn,tmax,cumulant,N);
            semilogy(eps_grid, real(delta_real),...
                'LineWidth',1.5,...
                'DisplayName',sprintf('%s',noise_list_names(noise_type)))
            hold on
            end
            
%             % the edgeworth epsilon-delta from given cumulants
%             fprintf('plot edgeworth eps-delta curves.\n')
%             for s=s_list
%                 fprintf('s=%d\n',s)
%                 delta_edgeworth = delta(a,eps_grid, @(t) edgeworth_CharFn(a,t,s,lambda,N),tmax,cumulant,N);
%                 semilogy(eps_grid, real(delta_edgeworth),...
%                     'LineWidth',1.5,...
%                     'DisplayName',sprintf('%s edgeworth, s=%d',noise_list_names(noise_type),s))
%                 hold on
%             end

%             % the edgeworth c.f. with our error bound, which gives an
%             % upper bound for the approximation
%             delta_edgeworth_3 = delta(a,eps, @(t) edgeworth_CharFn(a,t,3,lambda,N),tmax,cumulant,N);
%             delta_error = delta(a, eps, @(t) cf_error_our(t),tmax,cumulant,N);
%             semilogy(eps, abs(delta_edgeworth_3)+abs(delta_error),'--',...
%                 'LineWidth',1.5,...
%                 'DisplayName','Our Bound with s=3')
%             hold on         
        
        end

        % set the scale of plots, x-y ratio is 16:9
        plot_scale = 1;
        xlimit=plot_scale*800;
        ylimit=plot_scale*450;
        
        % the c.f. figure settings
        f=figure(1);
        f.Position = [0 0 xlimit ylimit];
        title(sprintf('Characteristic Functions, N=%d, q=%.3f, sen=%.2f, L1-cost=%.2f',...
        N,q,sens_list(sens_type),b),'FontSize',16)
%         title(sprintf('Characteristic Functions, N=%d, q=%.3f, sen=%.2f, L2-cost=%.2f',...
%         N,q,sens_list(sens_type),variance),'FontSize',16)
        legend('Location','southwest','FontSize',14)
        xlabel('t','FontSize',12);
        grid on

        % the eps-delta figure settings
        f=figure(2);
        f.Position = [0 0 xlimit ylimit];
        title(sprintf('Delta-Epsilon, N=%d, q=%.2f, sen=%.2f, L1-cost=%.2f',...
        N,q,sens_list(sens_type),b),'FontSize',16)
%         title(sprintf('Delta-Epsilon, N=%d, q=%.2f, sen=%.2f, L2-cost=%.2f',...
%         N,q,sens_list(sens_type),variance),'FontSize',16)
        legend('Location','southwest','FontSize',14)
        xlabel('epsilon','FontSize',12)
        ylabel('delta','FontSize',12)
        ylim([1e-5 1])
        grid on    
    end
end