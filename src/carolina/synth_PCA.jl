using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--opt1"
            help = "an option with an argument"
        "--opt2", "-o"
            help = "another option with an argument"
            arg_type = Int
            default = 0
        "--flag1"
            help = "an option without argument, i.e. a flag"
            action = :store_true
        "arg1"
            help = "a positional argument"
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

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
using TextAnalysis
using InvertedIndices
using StatsBase
using RecursiveArrayTools

function PCA(X,keepDims)
    Us = []
    Sigs = []
    Vts = []
    F = svd(X, full = false)
    U, Sig, Vt = F.U, F.S, F.Vt

    return [U[:,1:keepDims], Sig[1:keepDims], Vt[1:keepDims, :]]

end


function DTMCreate(true_data)
    rightDocs = []
    wrongDocs = []
    allDocs = []
    class = []

    for i in 1:length(true_data[:, 1])
        #First for loop collects items in field of interest
        if true_data[i, 2] == 1.0
            #Append to document collection
            push!(rightDocs, StripUnwanted((true_data[i, :])))
            push!(allDocs, StripUnwanted((true_data[i, :])))
            push!(class, (true_data[i, 2]))
        else
            #= Collects indeces not included in field of interest=#
            push!(wrongDocs, StripUnwanted((true_data[i, :])))
            push!(allDocs, StripUnwanted((true_data[i, :])))
            push!(class, (true_data[i, 2]))
        end
    end
    #Create corpus with cleaned string documents
    crps = Corpus(rightDocs)
    totalcrps = Corpus(allDocs)

    update_lexicon!(totalcrps)
    lex = lexicon(totalcrps)

    m = DtmHelper(totalcrps, lex)
    m = Vec2Mat(m)
    println("DTM has been created")
    return m
end

function DtmHelper(crps, lex)
    matrix = []
    t = [1]
    f = [0]
    for i in crps
        a = dtv(i, lex)
        a = vec(a)
        if author(i) == "1.0"
            push!(a, 1)
        else
            push!(a, 0)
        end

        push!(matrix, a)
    end

    return matrix
end

function Vec2Mat(v)
    VA = VectorOfArray(v)
    return(convert(Array, VA))
end

function StripUnwanted(row)
    sd = TextAnalysis.StringDocument(row[1])
    author!(sd, string(row[2]))
    return sd
end


function returnTesting(Vt, Sig, logit, test)
    beta = Vt'*inv(diagm(Sig))*coef(logit)

    rets = Matrix(test)[:,1:end - 1]*(beta)

    bin = 1 ./ (1 .+ exp.(-rets))

    return vec(bin)

end

n = parse(Int64, get(parsed_args, "arg1", 0 ))
param = Int(ceil(n/100))
seed = n%100
pc = [10,20,30,40,50,60,70,80,90]

datatot = CSV.read("src/cleanedData.csv", DataFrame)
Random.seed!(13)
datatot = filtration(datatot, " Cardiovascular / Pulmonary")
datatot = hcat(datatot[:,3], (datatot[:,1].== " Cardiovascular / Pulmonary"))
DTM = DTMCreate(datatot)
total_DTM = DataFrame(hcat(DTM', datatot[:,2]))

Random.seed!(13)
train, test = TrainTestSplit(total_DTM, .7)
complength = 80
trainmat = convert(Matrix{Float64}, train)
Us, Sigs, Vts = PCA(Matrix(trainmat)[:, 1:end - 1], complength)

dftrain = DataFrame(hcat(Us,train[:,end].==1))
    #Append Training Data with classifier

z=term(Symbol(:x, complength+1)) ~ term(0) + sum(term.(Symbol.(names(dftrain[:, Not(Symbol(:x, complength+1))]))))

logit = glm(z,dftrain, Bernoulli(), LogitLink())

classtest = test[:,end]
   #Calculating Error Rate
preds = returnTesting(Vts, Sigs, logit, test)

errors = 1-sum((preds .> .5) .== classtest)/length(preds)
    #Report the Error
    #=
JLD.save("Preds" *string(n)* ".jld", "val", preds)
JLD.save("Trues" *string(n)* ".jld", "val", classtest)
=#

preds = convert(Vector{Float64}, preds)
roc = MLBase.roc(classtest .== 1,preds)

TP = []
FP = []
for i in 1:length(roc)
    push!(TP,roc[i].tp/roc[i].p)
    push!(FP,roc[i].fp/roc[i].n)
end
push!(TP,0.0)
push!(FP,0.0)

Plots.plot(FP,TP, leg = false)
Plots.plot!(FP,TP, leg = false, seriestype = :scatter)

Plots.title!("PCA Regression")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")
xaxis!([0,1])
yaxis!([0,1])

JLD.save("Errors" * string(params) * "_" * string(seed) * ".jld", "val", errors)
