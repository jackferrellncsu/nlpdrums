include("../PCA.jl")

#This needs the train test split done
dtrain = CreateDTM(train, field)
dtest = CreateDTM(test,field)
DTMTrain = DataFrame(1.0*dtrain', :auto)
DTMTest = DataFrame(1.0*dtest', :auto)

param = 25 #This has to be as large as our largest wanted PCA
    #This line will take some time
    Us, Sigs, Vts = PCAVecs(Matrix(DTMtrain)[:, 1:end - 1], param)
    for ii in [25]

        dftrain = DataFrame(hcat(Us[ii],DTMtrain[:,end]), :auto)
        z=term(Symbol(:x, ii+1)) ~ term(0) + sum(term.(Symbol.(names(dftrain[:, Not(Symbol(:x, ii+1))]))))
        logit = glm(z,dftrain, Bernoulli(), LogitLink())
        beta = Vts[ii]'*inv(diagm(Sigs[ii]))*coef(logit)
        rets = Matrix(DTMtest)[:,1:end - 1]*hcat(beta)
        bin = 1 ./ (1 .+ exp.(-rets))
        ROC = MLBase.roc(DTMtest[:,end] .== 1,vec(bin), num)
        Plots.plot!(MLBase.false_positive_rate(ROC),MLBase.true_positive_rate.(ROC))
    end
