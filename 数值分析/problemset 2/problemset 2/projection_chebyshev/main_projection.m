%%-------------------------------------------------------------------------
% Solving Lucas'(1978) model with projection methods
%
% Numerical Methods in Finance
% Companion codes for problemsets
% Hong Lan, Nov. 2017
%--------------------------------------------------------------------------
%Housekeeping
clear all; close all; clc

%% Parameters
%--------------------------------------------------------------------------
par.beta  = 0.9;
par.gamma = 4.0;
par.mud   = 0.1;
par.rhod  = 0.95;
par.sigma = 0.1;

%% Grid
%--------------------------------------------------------------------------
%Construct grid
grid.order = 5;                           %order of chebyshev polynomial
grid.nnod  = grid.order + 1;              %number of chebyshev nodes
grid.d     = chebnode(grid.nnod);         %create chebyshev nodes 
grid.size  = grid.nnod;                   %size of grid

dmean = par.mud/(1-par.rhod);             %AR(1) mean
dstd  = par.sigma/sqrt((1-par.rhod^2));   %AR(1) std. dev.
grid.dmin  = dmean-3*dstd;                %min value of d(t)
grid.dmax  = dmean+3*dstd;                %max value of d(t)

%Gauss-Hermite nodes and weights
gh.size = 5; 
[gh.e,gh.w] = hernodes(gh.size); 

%% Projection for price
%--------------------------------------------------------------------------
%Initials
initials = 'fresh'; %Type 'fresh' for fresh initial values or 'previous' to load initial values from init.mat.
switch initials
    case 'fresh'
        init = zeros(grid.order+1,1); 
        init(grid.order+1,1) = par.beta/(1-par.beta); %Analytical solution for log utility.
    case 'previous'
        load init
end

%Minimization routine
options = optimset( 'Display',     'Iter',...
                    'MaxFunEvals', 1E5,...
                    'MaxIter',     1E5,...
                    'TolFun',      1E-10,...
                    'TolX',        1E-10             );
coef    = fminsearch(@(coef) errfunc(coef,par,grid,gh),init,options);
coef    = fminsearch(@(coef) errfunc(coef,par,grid,gh),coef,options); %Always run twice!!

%Save new initials
init = coef; 
save init init

%% Price
%--------------------------------------------------------------------------
%Compute price
pnow = chebpol(grid.order,grid.d)*coef;
%Plot price
figure(1);
plot(grid.d,pnow);
title('Price'), box off
xlabel('\it{d_{t}}'), ylabel('\it{p_{t}}','Rotation',0)

%% Projection for expected risk premium
%--------------------------------------------------------------------------
erp   = erpfunc(coef,par,grid,gh);      %This function calculates the expected risk premium at each grid point.
%COMPUTE THE COEFFICIENTS FOR x(t)
coefx = polyfit(grid.d,erp,grid.order);     

%% Simulation
%--------------------------------------------------------------------------
%Settings
sim.size = 500;                                     %Number of periods

%Stochastics
randn('state',519) ;
shock = par.sigma*randn(sim.size,1);                %Shock to dividend

%Simulation
sim.d = zeros(sim.size,1); 
sim.d(1) = dmean;
sim.x = zeros(sim.size,1);
for i = 2:sim.size,
    sim.d(i) = par.mud+par.rhod*sim.d(i-1)+shock(i);
    %COMPUTE THE SIMULATION OF x(t)
    sim.x(i) = 1/polyval(coefx,sim.d(i));
end

%Plot simulation
figure(2)
subplot(2,1,1), plot(1:sim.size,sim.d)
title('Dividend'), box off
xlabel('\it{t}'), ylabel('\it{d_{t}}','Rotation',0)
subplot(2,1,2), plot(1:sim.size,sim.x)
title('Expected risk premium'), box off
xlabel('\it{t}'), ylabel('\it{x_{t}}','Rotation',0)