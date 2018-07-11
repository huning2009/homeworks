clc;
clear all;
close all;

A = [2 1 5;4 4 -4;1 3 1];
b = [5;0;6];

x = A\b
[L,U,P] = lu(A)

[x,L,U,P] = nmgepp(A,b)