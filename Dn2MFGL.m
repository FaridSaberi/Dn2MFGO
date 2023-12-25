function [Hfinal,Vfinal] = Dn2MFGL(X,S,D,options)

%%%%%%%%%%%%%%%%%%%%
%% X: Data set in R_+^(m*n),  where m and n are the numbers of samples and features, respectively.
%% C: Centering matrix defined as C = I_n-(1/n)ee^T.
%% S: Similarity matrix in R_+^(n*n) associated with the data samples.
%% D: Degree matrix in R_+^(n*n) obtained from S.
%% L: Laplacian matrix in R^(n*n), L = D - S.
%% alpha, beta, nu: Regularization parameters.
%% r: Vector of size of each layer, r = [r_1, r_2,...,r_l] in which r_i is the size of each layer.
%% l: Number of layers.
%% maxiter: Maximum number of iterations

%% Parameters
r = options.r;
l = options.l;
alpha = options.alpha; 
beta = options.beta; 
nu = options.nu; 
maxiter = options.maxiter;

%% 
[m,n] = size(X);
onesn = ones(n,n);
E = (1/n)*ones(n,1)*(ones(n,1)');

%% Pretrain
[H,V] = Pretain(X,l,r);

%% Computing the L21 norm of X - H1H2...HlVl
Q = eye(m,m);
for i = 1:l
    Q = Q*H{i};
end
Z = X - Q*V{l};
P = L21(Z);



iter = 1;
while (iter <= maxiter)
    
    %% Initialization
    phi = cell(l+1,1);
    psi = cell(l+1,1);
    phi{1} = eye(m,m);
    psi{l+1} = eye(r(l),r(l));
    for i=1:l
        
        %% phi
        if i>1
            phi{i} = H{i-1};
            for j = i-2:-1:1
                phi{i} = H{j}*phi{i};
            end
        end
        
        %% psi
        if i <= l-1
            psi{i+1} = H{l};
            for jj = l-1:-1:i+1
                psi{i+1} = H{jj}*psi{i+1};
            end
        end
        
        %% Update Hi
        if i == 1
            A = psi{2}*V{l};
            numH = P*X*(A');
            denH = P*H{1}*A*(A');
            re1 = numH./max(denH,1e-10);
            re1 = nthroot(re1,2);
            H{1} = H{1}.*re1;
        else
            A = psi{i+1}*V{l};
            numH = (phi{i-1}')*P*X*(A');
            denH = (phi{i-1}')*P*phi{i-1}*H{i}*A*(A');
            re1 = numH./max(denH,1e-10);
            re1 = nthroot(re1,2);
            H{i} = H{i}.*re1;
        end
        
        %% Update phi
        if i>1
            phi{i} = phi{i}*H{i};
        else
            phi{i} = H{i};
        end
        
        %% Update Vl
        if i == l
            numV = (phi{l}')*P*X + alpha*V{l}*S + (beta+nu)*V{l};
            denV = (phi{l}')*P*phi{l}*V{l} + alpha*V{l}*D + beta*V{l}*E + nu*V{l}*onesn;
            re2 = numV./max(denV,1e-10);
            re2 = nthroot(re2,2);
            V{l} = V{l}.*re2;
        else
            numV = (phi{i}')*P*X + alpha*V{i}*S + (beta+nu)*V{i};
            denV = (phi{i}')*P*phi{i}*V{i} + alpha*V{i}*D + beta*V{i}*E + nu*V{i}*onesn;
            re2 = numV./max(denV,1e-10);
            re2 = nthroot(re2,2);
            V{i} = V{i}.*re2;
        end
    end
    
    %% Computing the L21 norm of X - H1H2...HlVl
    Q = eye(m,m);
    for i = 1:l
        Q = Q*H{i};
    end
    Z = X - Q*V{l};
    P = L21(Z);
    iter = iter + 1;
end

%% Computing the Basis Matrix H = H1H2...Hl
Hfinal = eye(m,m);
for i = 1:l
    Hfinal = Hfinal*H{i};
end

%% Compute the Representation Matrix Vl
Vfinal = V{l};