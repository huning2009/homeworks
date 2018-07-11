function erp = erpfunc(coef,par,grid,gh)

%Allocating memory
p    = zeros(grid.size,1);
er   = zeros(grid.size,1);
rf   = zeros(grid.size,1);
temp = zeros(gh.size,2);

%Expected rate of return
for i = 1:grid.size
    p(i,1) = chebpol(grid.order,grid.d(i,1))*coef;
    for j = 1:gh.size
        %COMPUTE d(t), i.e., dnow
        dnow = grid.d(i,1);
        dnew = par.mud+par.rhod*dnow+sqrt(2)*par.sigma*gh.e(j,1);
        %COMPUTE p(t+1), i.e., pnew
        pnew = chebpol(grid.order,dnew)*coef;
        qnew = par.beta*(dnew/dnow)^-par.gamma;
        temp(j,1) = gh.w(j,1)/sqrt(pi)*(dnew+pnew);
        temp(j,2) = gh.w(j,1)/sqrt(pi)*qnew;
    end
    er(i,1) = sum(temp(:,1))/p(i,1)-1;
    rf(i,1) = 1/(sum(temp(:,2)))-1;
end

%Expected risk premium
erp = er-rf;

end