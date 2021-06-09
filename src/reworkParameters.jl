using LinearAlgebra
function testModel(Vt, Sig, logit, test)
    beta = Vt'*inv(diagm(Sig))*coef(logit)

    rets = Matrix(test)[:,1:end - 1]*hcat(beta)

    bin = 1 ./ (1 .+ exp.(-rets)).>.5

    err = 1-sum(bin.==test[:,end])/length(test[:,end])

    return err

end
