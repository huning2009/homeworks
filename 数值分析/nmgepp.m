function [x,L,U,P] = nmgepp(A,b)
%% 初始条件
n = length(b);
L = zeros(n);
U = zeros(n);
P = eye(n);
%% LU分解
for i = 1:n-1
    max = abs(A(i,i));
    k = i; 
    for j = i+1:n 
        if  max < abs(A(j,i)) 
            max = abs(A(j,i));
            k = j;
        end
    end
    if (k~=i) 
        TempA = A(i,:);
        A(i,:) = A(k,:);
        A(k,:) = TempA;
        
        TempP = P(i,:);
        P(i,:) = P(k,:);
        P(k,:) = TempP; 
        
        TempL = L(i,:);
        L(i,:) = L(k,:);
        L(k,:) = TempL; 
        
        Tempb = b(i);
        b(i) = b(k);
        b(k) = Tempb;
    end
    
    for j = i+1:n
        L(j,i) = A(j,i)/A(i,i);
        A(j,:) = A(j,:) - A(j,i)/A(i,i)*A(i,:);
        b(j) = b(j)  - A(j,i)/A(i,i)*b(i);
    end
end
for i = 1:n
    L(i,i) = 1;
end
U = A;
for i = 2:n
    b(i) = b(i) - L(i,1:i-1)*b(1:i-1,1);
end
%% 回代 
x(n,1) = b(n)/U(n,n);
for i = n-1:-1:1
    x(i,1) = (b(i)-U(i,i+1:n)*x(i+1:n,1))./U(i,i);
end
