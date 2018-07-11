%% ---------------------------------------------------------------
% Bayesian Inferences 
%
% Numerical Methods in Finance
% Companion codes for problemsets
% Hong Lan, Nov. 2017
%--------------------------------------------------------------------------
%Housekeeping
clear all; close all; clc

%% Load Dataset
%--------------------------------------------------------------------------
load Y;

%% Maximum Likelihood Estimation
%--------------------------------------------------------------------------     
%Algorithm parameters              
 options = optimoptions( @fminunc,...
                         'Algorithm',   'quasi-newton',...
                         'MaxFunEvals', 1E5,...
                         'MaxIter',     1E5,...
                         'TolFun',      1E-10,...
                         'TolX',        1E-10,...
                         'Display',     'iter'                    );
                    
%Maximum likelihood estimation
 %Initial guess:mean, variance and degree of freedom
  theta_init = [ 8  3  20 ]';
 %Maximum likelihood estimation using MATLAB solver fminunc
  [theta_mle,~,~,~,~,hessian] = fminunc(@(param) llt(Y,param) ,theta_init,options );

%Pring results
disp('Simulated mean:')
disp(mean(Y))
disp('Simulated standard deviation:')
disp(std(Y)) 
disp('Estimated mean')
mu_mle = theta_mle(1);
disp(mu_mle)
disp('Estimated standard deviation')
var_mle = theta_mle(2);
disp(sqrt(var_mle))
disp('Estimated degree of freedom')
nu_mle = theta_mle(3);
disp(nu_mle)
disp('Standard error of the estimated mean')
s_mu = llt(Y,theta_mle);
disp(s_mu)

%% Compute Posterior Using Importance Sampling
%--------------------------------------------------------------------------
%Number of importance sampling draws
nis = 5000;

%Sample from the normal proposal density, note the mean of this proposal
%density is mu_mle and the variance is (3*s_mu)^2
mu_prop = mu_mle + sqrt( (3*s_mu)^2 ) * randn(nis,1);

%Compute the (log) proposal density
prop = lpdfn(mu_prop,theta_mle);

%% Compute the prior density
%Set the parameters of the normal prior
mu0  = 0;    % mean
var0 = 0.1;  % variance

%Compute the normal prior density
prior = mu0+var0*randn(nis,1);
 
%% Compute a posterior kernel
%Allocating memory
f     = zeros(nis,1);             % likelihood
posterior_kernel = zeros(nis,1);  % a kernel of the posterior

%Importance sampling
for i = 1:nis
      
    %Make the code cooler!  
    progressbar(i/nis);
    
    %Compute the log likelihood
    f(i) = 'YOUR CODE'; 
     
    %A log posterior kernel
    posterior_kernel(i) = 'YOUR CODE'; 

end

%% Compute the importance weight
%Compute the importantce weight in logs
wlog = 'YOUR CODE';

%Compute the importance weight
wlog  = wlog-max(posterior_kernel-prop);
w     = exp(wlog); 

%Compute the standardized weight
omega = w/sum(w); 

%% Compute posterior moments
%--------------------------------------------------------------------------
mu_posterior = mu_prop'*omega;
disp(['Posterior mean: ' num2str(mu_posterior)])
var_posterior = ((mu_prop-mu_posterior).^2)'*omega;
disp(['Posterior Standard Deviations: ' num2str(sqrt(var_posterior))])

%% Plot results
%--------------------------------------------------------------------------
figure;

%Prior
range = mu0-4*sqrt(var0):0.001:mu0+4*sqrt(var0);
value = normpdf(range,mu0,sqrt(var0));
plot(range,value/max(value));
hold on;

%Posterior kernel
[kd, pts] = ksdensity(mu_prop);
plot(pts,kd/max(kd));

%Plot settings
legend('prior','posterior kernel');
xlabel({'$\mu$'},'Interpreter','latex');
ylabel('(kernel) density');
title('Bayesian Inference');


