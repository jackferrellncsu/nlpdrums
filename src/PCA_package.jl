include("DTMCreation.jl")
include("data_cleaning.jl")
include("PCA.jl")
using MultivariateStats
using Lathe
using DataFrames
using Plots
using GLM
using StatsBase
using MLBase
using ROCAnalysis
using CSV

data = importClean()
dtmi = CreateDTM(data, " Cardiovascular / Pulmonary")

class = dtmi[end, :]
dtmi = dtmi[1:end-1, :]
floatDtmi = 1.0*dtmi

M = fit(PCA, floatDtmi; maxoutdim = 20)
reducedMat = transform(M, floatDtmi)

newRedMat = hcat(reducedMat', (1.0*class))

df = DataFrame(newRedMat, :auto)

using Lathe.preprocess: TrainTestSplit
train, test = TrainTestSplit(df, .9);

# fm = @formula(term(:x101) ~ sum(term.(Symbol.(names(df[:, Not(:x101)])))))
logit = glm(term(:x21) ~ sum(term.(Symbol.(names(df[:, Not(:x21)])))),
                                            train, Binomial(), ProbitLink())

prediction = GLM.predict(logit,test)

pred = Vector{Float64}()

for i in prediction
    push!(pred,i)
end

rocnums = MLBase.roc(test[:,end] .== 1,(pred))

TP = []
FP = []
base = []
area = 0
for i in 1:length(rocnums)
    push!(TP,rocnums[i].tp/rocnums[i].p)
    push!(FP,rocnums[i].fp/rocnums[i].n)
    area += TP[end]*1/length(rocnums)
end

#sum(prediction .!= test[:,end])/size(test)[1]

plot(FP,TP)
plot!((-1:101)./100, (-1:101)./100, leg = false)
