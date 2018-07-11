clc;
clear all;
close all;
A = [1 2 0 0;-1 2 3 0;0 2 4 5;0 0 -3 2];
B = [1;-2;3;4];
n = length(B);
for i = 1:n
    if i<n
        a(i) = A(i+1,i);
        c(i) = A(i,i+1);
    end
        b(i) = A(i,i);
end
d = B;

x = nmtridsolve(a,b,c,d)

x = A\B