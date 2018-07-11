function xd = scaldown(x,xmin,xmax)
%--------------------------------------------------------------------------
% xd = scaldown(x,xmin,xmax);
%
% Purpose:
% Linearly scale a variable from [min,xmax] to [-1,1], where x, xmin, xmax
% can be vectors as long as they are all of the same dimension
%--------------------------------------------------------------------------
[r,c]   = size(x);
a       = 2*ones(r,c) ./ ( xmax - xmin );
b       = ones(r,c) - 2 * xmax ./ ( xmax - xmin );

xd      = b + a .* x;

