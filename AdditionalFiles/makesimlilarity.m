function [L,D,S] = makesimlilarity(X,K)
options = [];
options.NeighborMode = 'KNN';
options.k = K;
options.WeightMode = 'Binary';
S = constructW(X,options);
D = diag(sum(S));
L = D - S;
end