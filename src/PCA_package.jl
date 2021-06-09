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
for i in 1:100
    println(i)
    U, Sig, Vt = PCA(Matrix(train)[:, 1:end - 1], i)

    dftrain = DataFrame(hcat(U,train[:,end]), :auto)

    logit = glm(term(Symbol(:x, i+1)) ~ sum(term.(Symbol.(names(dftrain[:, Not(Symbol(:x, i+1))])))),
                                                dftrain, Binomial(), ProbitLink())
    push!(errors,testModel(Vt, Sig, logit, test))
end

plot(1:100,errors)

errors2 = []
for B in 1:100
    for bb in 1:1000
        train, test = TrainTestSplit(df, .9)

        MultivariateStats.fit(MultivariateStats.PCA, Matrix(train)[:, 1:end-1], B)

        model = glm()
    end
end
df_2 = DataFrame(1.0*dtmi, :auto)
train, test = TrainTestSplit(df_2, .9)

trainPreds = train[1:end - 1, :]
trainOutcomes = convert(Array{Any}, train[end, :])

M = MultivariateStats.fit(MultivariateStats.PCA, Matrix(trainPreds); maxoutdim = 100)
pcaX = MultivariateStats.transform(M, Matrix(trainPreds))

pca_trainDf = DataFrame(hcat(pcaX', trainOutcomes))

model = glm(term(Symbol(:x, 101)) ~ sum(term.(Symbol.(names(pca_trainDf[:, Not(Symbol(:x, 101))])))),
                                            pca_trainDf, Binomial(), ProbitLink())

model = glm(term(Symbol(:x, 101)) ~ sum(term.(Symbol.(names(pca_trainDf, Not(:x101))))),
                                            pca_trainDf, Binomial(), LogitLink())
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
