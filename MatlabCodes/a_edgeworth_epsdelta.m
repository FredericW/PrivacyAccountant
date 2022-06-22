% ----------------------------------------------
% opt_03_main.m
% last updated: 10/25/2021
% ----------------------------------------------

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

% ----------------------------------------------
% KEY VARIABLES
% ----------------------------------------------

% dpsgd=["gaussian","laplace","cactus"];% 1:gaussian 2:laplace 3: cactus
dpsgd=["gaussian","laplace","cactus"];% 1:gaussian 2:laplace 3: cactus
l2_norm_limit=[1];
noise_multiplier=[2];
s = 3; % number of cumulants in approx.

N = 1000; % composition
q = 1/100; % subsampling rate

a = 0.01; %  avoid the pole

fprintf('N=%d, q=%.3f, s=%d\n', N, q, s)

% ----------------------------------------------

for norm_type=1:length(l2_norm_limit)
    
    for noise_type= 1:length(noise_multiplier)
        
        variance=(l2_norm_limit(norm_type)*noise_multiplier(noise_type))^2;
        
        d = l2_norm_limit(norm_type); %shift or sensitivity
        
        fprintf('sen=%.2f\n',d)
        
        for dpsgd_type=2
            
            if dpsgd_type==3
                filename = sprintf('cactus_pdf_d%.1f_v%.2f.mat',d,variance);
                load(filename,'xmax');
            else
                xmax  =(l2_norm_limit*noise_multiplier(noise_type))*10;
            end
            
            fprintf('xmax=%.1f\n',xmax)
            
            fprintf('var=%.2f\n', variance)
            
            % ----------------------------------------------
            
            switch dpsgd_type
                case 1
                    f = @(x) 1/ sqrt(2*pi)/sqrt(variance)...
                        * exp(-(x).^2/(2*sqrt(variance)^2));
                case 2
                    f = @(x) 1/2/sqrt(variance/2)...
                        * exp(-abs(x)/sqrt(variance/2));
                otherwise
                    f = @(x) pdf_cactus(x,d,variance);
            end
            
            LogLoss = @(x) log(f(x)./(q*f(x+d)+(1-q)*f(x)));
            
            
            % ----------------------------------------------
            
            disp('Parameters.')
            
            [cumulant, lambda, rho]=parameters(f,LogLoss,s,xmax);
            
            % ----------------------------------------------
            
            disp('Characteristic functions.')
            
            real_CharFn = @(t) sum(chebfun(@(x) f(x)...
                .*exp((a+1i.*t).*(LogLoss(x)-cumulant(1))/sqrt(N*cumulant(2))),...
                [-xmax,xmax],'splitting','on')).^N;
            
            % ----------------------------------------------
            
            disp('Error estimates.')
            
            % This is our upper bound regading the real error, details see write-up
            
            T_our = sqrt(N)/s/rho(s)^(1/s); % integral limits
            
            fprintf('T_our=%.4f\n',T_our);
            
            cc = 3*exp(1)*rho(s)^(1/s)/2/s;
            cf_error_our =...
                @(t) s^(s-2)/T_our^(s-2) * ...
                ( (cc^(s-2).*abs(t).^(2*(s-2))-cc.*abs(t).^(s-1))./ (cc.*abs(t)-1)...
                + cc^(s-2).*abs(t).^(2*(s-2))/factorial(s-2) )...
                .* exp(-t.^2/2);
            
            
            % This is Kolmogorov's upper bound from his book.
            
            T_kol = sqrt(N)/8/s/rho(s)^(3/s);
            
            fprintf('T_kol=%.4f\n', T_kol);
            
            cc_1= @(t) 3*(s-3)/s^2 * exp(s/8)* s^(s-2)...
                * (rho(s)^(3/s)/sqrt(N))^(s-2)...
                * (abs(t).^s + abs(t).^(3*s-6));
            cc_2 = @(t) 3^(s-2)*(rho(s)^(1/s).*abs(t)).^(3*s-6)...
                *(1/sqrt(N))^(s-2).*exp(s/8+t.^2/4);
            cf_error_kol =...
                @(t) abs(exp(-t.^2/2).*(cc_1(t)+cc_2(t)));
            
            tmax=T_our;
            
            % ----------------------------------------------
            
            
            x=linspace(-xmax,xmax,100);
            
            figure
            plot(x,LogLoss(x),'DisplayName','LogLoss')
            grid on
            title(sprintf('%s, q=%.2f, sen=%.2f, var=%.2f',...
                dpsgd(dpsgd_type),q,l2_norm_limit(norm_type),variance))
            legend('Location','best')
            xlabel('x','FontSize',10);
            
            
            t=linspace(0,tmax,100);
            cf_real=real_CharFn(t);
            cf_edgeworth = edgeworth_CharFn(a,t,s,lambda);
            error_real = cf_real-cf_edgeworth;
            error_our = cf_error_our(t);
            %             error_kol = cf_error_kol(t);
            
            figure
            
            semilogy(t, abs(cf_real),...
                'LineWidth',1.5,'DisplayName','Real C.F.')
            hold on
            semilogy(t, cf_edgeworth,'--',...
                'LineWidth',1.5,'DisplayName',sprintf('Edgeworth C.F. s=%d',s))
            hold on
            semilogy(t, error_our+cf_edgeworth,'.',...
                'LineWidth',1.5,'DisplayName','Our Bound with s=%d',s)
            grid on
            legend('Location','southwest')
            xlabel('t','FontSize',12);
            
            figure
            
            semilogy(t, abs(error_real),...
                'LineWidth',1.5,'DisplayName','Real Error')
            hold on
            semilogy(t, error_our,'--',...
                'LineWidth',1.5,'DisplayName',sprintf('Our Bound with s=%d',s))
            hold on
            semilogy(t, error_kol,'--','DisplayName','Kol Bound')
            grid on
            title(sprintf('%s, N=%d, q=%f, sen=%.2f, var=%.2f',...
                dpsgd(dpsgd_type),N,q,l2_norm_limit(norm_type),variance))
            legend('Location','best')
            xlabel('t','FontSize',10);
            ylim([10^(-10) 10^1])
            
            
            % ----------------------------------------------
            
            disp('Epsilon-Delta.')
            
            eps = linspace(0,8*sqrt(N)*cumulant(1)/sqrt(cumulant(2)),25); % range of epsilon
            
            delta_edgeworth = delta(eps, @(t) edgeworth_CharFn(a,t,s,lambda),tmax);
            
            if dpsgd_type==3
                delta_real = delta_edgeworth;
            else
                delta_real =  delta(eps, real_CharFn,tmax);
            end
            
            % ----------------------------------------------
            
            disp('Save results.')
            
            filename = sprintf('d_%s_s%d_d%1.1f_v%1.2f_N%d.mat',...
                dpsgd(dpsgd_type),s,d,variance,N);
            save(filename,...
                't','cf_real','cf_edgeworth',...
                'eps','delta_real','delta_edgeworth');
        end
    end
end
