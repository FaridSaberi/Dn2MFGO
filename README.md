# Deep Nonnegative Matrix Factorization with Joint Global and Local Structure Preservation 

# Abstract

Deep Non-Negative Matrix Factorization (DNMF) methods provide an efficient low-dimensional representation of given data through their layered architecture. A limitation of such methods is that they cannot effectively preserve the local and global geometric structures of the data in each layer. Consequently, a significant amount of the geometrical information within the data, present in each layer of the employed deep framework, can be overlooked by the model. This can lead to an information loss and a subsequent drop in performance. In this paper, we propose a novel deep non-negative matrix factorization method, Deep Non-Negative Matrix Factorization with Joint Global and Local Structure Preservation (dubbed Dn2MFGL), that ensures the preservation of both global and local structures within the data space. Dn2MFGL performs representation learning through a sequential embedding procedure which involves both the global data structure by accounting for the data variance, and the local data relationships by utilizing information from neighboring data points. Moreover, a regularization term that promotes sparsity by utilizing the concept of the inner product is applied to the matrices representing the lower dimensions. This aims to retain the fundamental data structure while discarding less crucial features. Simultaneously, the residual matrix of Dn2MFGL is subjected to the L21 norm, which ensures the robustness of the model against noisy data samples. An effective and multiplicative updating process also facilitates Dn2MFGL in solving the employed objective function. The clustering performance of the proposed deep NMF method is explored across various benchmarks of face datasets. The results point to Dn2MFGL outperforming several existing classical and state-of-the-art NMF methods.

This repository provides an implementation for Dn2MFGL as described in the paper:

Farid Saberi-Movahed, Bitasta Biswas, Prayag Tiwari, Jens Lehmann, Sahar Vahdati, Deep Nonnegative Matrix Factorization with Joint Global and Local Structure Preservation, Under Review, 2023.

# Requirements

The codebase has been implemented in Matlab 2021. To use it, please run the file main.m.
