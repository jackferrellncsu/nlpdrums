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


errors = []
params,runs = [100,100]
Errors = Matrix(undef, params,runs)

for i in 1:runs
    train, test = TrainTestSplit(df, .9);
    println(i)
    Us, Sigs, Vts = PCAVecs(Matrix(train)[:, 1:end - 1], params)
    for ii in 1:params
        dftrain = DataFrame(hcat(Us[ii],train[:,end]), :auto)
        z=term(Symbol(:x, ii+1)) ~ term(0) + sum(term.(Symbol.(names(dftrain[:, Not(Symbol(:x, ii+1))]))))

        logit = glm(z,dftrain, Bernoulli(), LogitLink())

        Errors[ii,i] = testModel(Vts[ii], Sigs[ii], logit, test)
    end
end

plot(1:100,errors)

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
