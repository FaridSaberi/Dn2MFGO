function [H,V]=Pretain(X,l,r)
maxiter = 100;
Z = X;
H = cell(l,1);
V = cell(l,1);
for i=1:l
    [m,n] = size(Z);
    [H{i},V{i}] = stanNMF(Z,r(i),m,n,maxiter);
    Z = V{i};
end


