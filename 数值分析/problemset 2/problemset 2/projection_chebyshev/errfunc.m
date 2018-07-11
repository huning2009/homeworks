function ssr = errfunc(coef,par,grid,gh)

%Allocating memory
lhs  = zeros(grid.size,1);
rhs  = zeros(grid.size,1);
temp = zeros(gh.size,1);

%Double for loop
for i = 1:grid.size,
    %CALCULATE HERE THE LHS OF THE EULER EQUATION  
    lhs(i,1) = chebpol(grid.order,grid.d(i,1))*coef;
    for j = 1:gh.size,
     %CALCULATE HERE THE RHS OF THE EULER EQUATION FOR EACH GAUSSIAN
     %HERMITE NODE
       %COMPUTE d(t), i.e., dnow
        dnow = grid.d(i,1);    
        dnew = par.mud+par.rhod*dnow+sqrt(2)*par.sigma*gh.e(j,1);
       %COMPUTE p(t+1), i.e., pnew
        pnew = chebpol(grid.order,dnew)*coef;
        qnew = par.beta*(dnew/dnow)^-par.gamma;
        temp(j,1) = gh.w(j,1)/sqrt(pi)*qnew*(dnew+pnew);
    end
    %CALCULATE HERE THE RHS OF THE EULER EQUATION BY COMBINING THE ELEMENTS
    %OF TEMP. (NOTE THAT THE ELEMENTS OF TEMP WILL BE OVERWRITTEN BY THE
    %CALCULATIONS FOR THE NEXT GRID POINT.)
    rhs(i,1) = sum(temp);
end

%Sum of squared Euler equation errors
%CALCULATE HERE THE SUM OF SQUARED EULER EQUATION ERRORS
ssr = norm(lhs-rhs);

end