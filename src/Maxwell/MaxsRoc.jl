train, test = TrainTestSplit(df, .9);
param = 500
num = 50
Rocs = zeros(3,2,num)
    Us, Sigs, Vts = PCAVecs(Matrix(train)[:, 1:end - 1], param)
    for ii in [5,25,500]
        println(ii)
        dftrain = DataFrame(hcat(Us[ii],train[:,end]), :auto)
        z=term(Symbol(:x, ii+1)) ~ term(0) + sum(term.(Symbol.(names(dftrain[:, Not(Symbol(:x, ii+1))]))))
        logit = glm(z,dftrain, Bernoulli(), LogitLink())
        beta = Vts[ii]'*inv(diagm(Sigs[ii]))*coef(logit)
        rets = Matrix(test)[:,1:end - 1]*hcat(beta)
        bin = 1 ./ (1 .+ exp.(-rets))
        rocnums = MLBase.roc(test[:,end] .== 1,vec(bin), num)
        for i in 1:(num-1)
            if ii == 5
                Rocs[1,1,i] = rocnums[i].tp/rocnums[i].p
                Rocs[1,2,i] = rocnums[i].fp/rocnums[i].n
            elseif ii == 25
                Rocs[2,1,i] = rocnums[i].tp/rocnums[i].p
                Rocs[2,2,i] = rocnums[i].fp/rocnums[i].n
            else
                Rocs[3,1,i] = rocnums[i].tp/rocnums[i].p
                Rocs[3,2,i] = rocnums[i].fp/rocnums[i].n
            end
        end
    end
