%% An implementation of Deep Nonnegative Matrix Factorization with Joint Global and Local Structure Preservation
%% Farid Saberi Movahed
%% Email: f.saberimovahed@kgut.ac.ir & fdsaberi@gmail.com
%% 12-1-2023
%%
%% Dn2MFGL: Proposed Method 
%%
%% Inputs
%% X: Data set in R_+^(m*n),  where m and n are the numbers of samples and features, respectively.
%% C: Centering matrix defined as C = I_n-(1/n)ee^T.
%% S: Similarity matrix associated with the data samples.
%% D: Degree matrix obtained from S.
%% L: Laplacian matrix, L = D - S.
%% alpha, beta, nu: Regularization parameters.
%% r: Vector of size of each layer, r = [r_1, r_2,...,r_l] in which r_i is the size of each layer.
%% l: Number of layers.
%% maxiter: Maximum number of iterations
%%
%% Outputs
%% H: Basis Matrix defined as H = H_1*H_2*...*H_l
%% V: Coefficient Representation matrices obtained from the each layer 
clc
clear
close all
format shortG
addpath('./AdditionalFiles');

%% Input Data 
load('UMIST.mat')
data = NormalizeFea(X,1);   %% If you want to normalize data, please uncomment this line
X = data';


%% Parameters
r1 = 150; %% Size of the first latent space. Suggested range: {80,100,150,200}
r2 = 10; %% Size of the second latent space. Suggested range: {10,20,30,40,50,60,70}
options.r = [r1,r2];
options.l = length(options.r);
options.alpha = 1; %% This parameter needs to be tuned. Suggested range: {10^-8,10^-6,10^-4,10^-2,1,10^2,10^4,10^6,10^8}
options.beta = 1; %% This parameter needs to be tuned. Suggested range: {10^-8,10^-6,10^-4,10^-2,1,10^2,10^4,10^6,10^8}
options.nu = 1; %% This parameter needs to be tuned. Suggested range: {10^-8,10^-6,10^-4,10^-2,1,10^2,10^4,10^6,10^8}
options.maxiter = 100;


%% Laplacian, Similarity and Degree Matrices
K = 5; %% Size of neighbors. This parameter needs to be tuned. Suggested range: {2,5,10,15,20}
[L,D,S] = makesimlilarity(X',K);


%% Dn2MFGL method
[H,V] = Dn2MFGL(X,S,D,options);
