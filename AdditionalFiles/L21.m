function D = L21(Z)
z=4*max(diag(Z*(Z')),1e-10);
D = diag(sqrt(1./z));
end

