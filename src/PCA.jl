#X, Your Matrix, RemoveDims, # of removed dimensions
using LinearAlgebra
using Statistics
function PCA(X,keepDims)
    removeDims = size(X)[1] - keepDims
    for i in 1:size(X)[1]
        X[i,:] = (X[i,:] .- mean(X[i,:])) ./ std(X[i,:])
    end
    S = svd(X, full = false)
    U, Sig, Vt = S.U, S.S, S.Vt
    Sig = Sig[1:end - removeDims]
    U = U[:,1:end - removeDims]
    Vt = Vt[1:end - removeDims, :]

    return [U, Sig, Vt]

end
