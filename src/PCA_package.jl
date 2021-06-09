include("DTMCreation.jl")
include("data_cleaning.jl")
include("PCA.jl")
include("reworkParameters.jl")
using MultivariateStats
using Lathe
using DataFrames
using Plots
using GLM
using StatsBase
using MLBase
using CSV
using Statistics
using Lathe.preprocess: TrainTestSplit

data = importClean()
dtmi = CreateDTM(data, " Cardiovascular / Pulmonary")
df = DataFrame(1.0*dtmi', :auto)
train, test = TrainTestSplit(df, .9)

errors = []
for i in 1:1
    println(i)
    U, Sig, Vt = PCA(Matrix(train)[:, 1:end - 1], i)

    dftrain = DataFrame(hcat(U,train[:,end]), :auto)
    print(typeof(dftrain))
    logit = glm(term(Symbol(:x, i+1)) ~ sum(term.(Symbol.(names(dftrain[:, Not(Symbol(:x, i+1))])))),
                                                dftrain, Binomial(), ProbitLink())

    push!(errors,testModel(Vt, Sig, logit, test))
end

plot(1:100,errors)

errors2 = []
for B in 1:1
    for bb in 1:10
        train, test = TrainTestSplit(df2, .9)

        pcaTrainInput = DataFrame(Matrix(train[:, 1:end - 1])')

        M = MultivariateStats.fit(MultivariateStats.PCA, Matrix(pcaTrainInput); maxoutdim = 20)
        pcaX = MultivariateStats.transform(M, Matrix(pcaTrainInput))

        logitTrainInput = DataFrame(hcat(pcaX', train[:, end]))

        form = term(Symbol(:x, 20+1)) ~ sum(term.(Symbol.(names(logitTrainInput[:, Not(Symbol(:x, 20+1))]))))
        model = glm(form, logitTrainInput, Bernoulli(), LogitLink())

    end
end
#PCA needs columns to be observations
#Logit needs columns to be predictors
df2 = DataFrame(dtmi*1.0)
df2 = DataFrame(Matrix(df2)')

train, test = TrainTestSplit(df2, .9)

pcaTrainInput = DataFrame(Matrix(train[:, 1:end - 1])')

M = MultivariateStats.fit(MultivariateStats.PCA, Matrix(pcaTrainInput); maxoutdim = 20)
pcaX = MultivariateStats.transform(M, Matrix(pcaTrainInput))

logitTrainInput = DataFrame(hcat(pcaX', train[:, end]))

form = term(Symbol(:x, 20+1)) ~ term(0) + sum(term.(Symbol.(names(logitTrainInput[:, Not(Symbol(:x, 20+1))]))))
model = glm(form, logitTrainInput, Bernoulli(), LogitLink())

beta = coef(model)

beta_hat = beta\M.proj'
#=
rocnums = MLBase.roc(test[:,end] .== 1,vec(rets), 50)

TP = []
FP = []
for i in 1:length(rocnums)
    push!(TP,rocnums[i].tp/rocnums[i].p)
    push!(FP,rocnums[i].fp/rocnums[i].n)
end

plot(FP,TP)
plot!((0:100)./100, (0:100)./100, leg = false)
=#
