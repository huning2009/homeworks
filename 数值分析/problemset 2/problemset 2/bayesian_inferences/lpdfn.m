function lpdfn = lpdfn(y,param)
%--------------------------------------------------------------------------
% lpdfn = lpdfn(y,param) computes the log density of a normal distribution
%
% y: observations
% param: parameters of the normal distribution, in order of [mu;variance]
%--------------------------------------------------------------------------

lpdfn = -log(sqrt(2*pi*param(2)))-0.5*( (y-param(1)).*(y-param(1))./param(2) );