include("../DTMCreation.jl")
include("../data_cleaning.jl")
include("../PCA.jl")
include("../reworkParameters.jl")
include("../emi/DTM_convolution.jl")
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
using Flux

true_data = CSV.read("wordy.csv", DataFrame)
true_data = true_data[1:1000, :]
DTM = CreateDTM(true_data)
total_DTM = DataFrame(DTM')

errors = []
parameters,runs = [25,100]
Errors = Matrix(undef, parameters,runs)

for i in 1:runs
    train, test = TrainTestSplit(total_DTM, .9);
    println(i)
    Us, Sigs, Vts = PCAVecs(Matrix(train)[:, 1:end - 1], parameters)
    for ii in 1:parameters
        dftrain = DataFrame(hcat(Us[ii],train[:,end]))
        z=term(Symbol(:x, ii+1)) ~ term(0) + sum(term.(Symbol.(names(dftrain[:, Not(Symbol(:x, ii+1))]))))

        logit = glm(z,dftrain, Bernoulli(), LogitLink())

        Errors[ii,i] = testModel(Vts[ii], Sigs[ii], logit, test)
    end
end

for i in 1:24
    push!(errors, mean(Errors[i,:]))
end

P = Plots.plot(1:50,errors, leg = false, title = "Cardiovascular Cross Validation")
xlabel!("# of Principal Components")
ylabel!("Validation Error")


rocnums = MLBase.roc(test[:,end] .== 1,vec(rets), 50)

TP = []
FP = []
for i in 1:length(rocnums)
    push!(TP,rocnums[i].tp/rocnums[i].p)
    push!(FP,rocnums[i].fp/rocnums[i].n)
end

plot(FP,TP)
plot!((0:100)./100, (0:100)./100, leg = false)
