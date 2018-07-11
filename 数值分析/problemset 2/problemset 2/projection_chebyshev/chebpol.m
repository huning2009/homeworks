function x=chebpol(order,p)
%--------------------------------------------------------------------------
% x = chebpol(order,p);
% Purpose:
% Create an order-th order Chebyshev Polynomial 
%
% If p	is a vector then the polynomial will be created based on that
% -------------------------------------------------------------------------
[r , ~]	= size(p);
x = ones(r,order+1);	
x(:,2)	= p;
if order >= 2;
   for i = 3:order+1;
       x(:,i) = 2.*p.*x(:,i-1)-x(:,i-2);
   end
end


