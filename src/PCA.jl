#X, Your Matrix, RemoveDims, # of removed dimensions
using LinearAlgebra
using Statistics
function PCAVecs(X,keepDims)
    Us = []
    Sigs = []
    Vts = []
    S = svd(X, full = false)
    U, Sig, Vt = S.U, S.S, S.Vt
    for i in 1:keepDims
        push!(Sigs,Sig[1:i])
        push!(Us,U[:,1:i])
        push!(Vts,Vt[1:i, :])
    end

    return [Us, Sigs, Vts]

end
