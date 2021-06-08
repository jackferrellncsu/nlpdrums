#X, Your Matrix, RemoveDims, # of removed dimensions
using LinearAlgebra
function PCA(X,removeDims)

    U, Sig, Vt = svd(X, full = true).U, svd(X,full = true).S, svd(X,full = true).Vt
        Sig = Sig[1:end - removeDims]
        U = U[:,1:end - removeDims]
        Vt = Vt[1:end - removeDims, :]

        return U * diagm(Sig) * Vt
end
