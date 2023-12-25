function [H,V] = stanNMF(X,k,m,n,maxiter)
%% Initialization
H=abs(randn(m,k));
V=abs(randn(k,n));
iter=1;
E(iter)=(norm(X-H*V,'fro')^2)/n;
err = 1;
while (iter<=maxiter && err >= 10^-6)
    %%Update H
    numH=X*(V');
    denH=H*V*(V');
    re1=numH./max(denH,1e-10);
    H = H.*re1;  
    %%Update V
    numV=(H')*X;
    denV=(H')*H*V;
    re2=numV./max(denV,1e-10);
    V = V.*re2;  
    iter = iter+1;
    E(iter)=(norm(X-H*V,'fro')^2)/n;
    err = (E(iter-1)-E(iter))/max(1,E(iter-1));
end
end