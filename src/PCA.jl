#X, Your Matrix, RemoveDims, # of removed dimensions
function PCA(X,removeDims)

    U, Sig, Vt = svd(X).U, svd(X).S, svd(X).Vt

        index = argmin(EigenValues)
        Sig = Sig[1:end - removeDims]
        U = U[:,1:end - removeDims]
        Vt = Vt[1:end - removeDims, :]

        return U * diagm(Sig) * Vt
    end

end
