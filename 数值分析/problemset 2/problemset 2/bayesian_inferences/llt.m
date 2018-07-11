function llt = llt(y,param)
%--------------------------------------------------------------------------
% llt = llt(y,param) computes the log likelihood of the t distribution
%
% y: observations
% param: parameters of the t distribution, in order of [mu;variance;df]
%--------------------------------------------------------------------------

N = max(size(y));
if N~=size(y,1)
    y = y';
end

mu = param(1);
variance = param(2);
df = param(3);

if(variance<0 || df<2)
    llt = -1e10;
else
    llt = sum(-(df+1)*0.5*log(1+(y-mu).*(y-mu)./(variance*df)));
    llt = (llt + N*(gammaln((df+1)*0.5) - 0.5*log(df*pi) -gammaln(0.5*df)) -0.5*N*log(variance));
end
