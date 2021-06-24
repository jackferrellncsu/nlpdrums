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

using Word2Vec
using DataFrames
using GLM
using StatsBase
using CSV
using Statistics
using Languages
using Lathe.preprocess: TrainTestSplit
using LinearAlgebra
using JLD
using Random

vecLength1 = 15
vecLength2 = 0
#Max is 50
window1 = 50

M = wordvectors("vectors0.txt", normalize = false)

function formulateText(model, script)
   words = split(script, " ")
   vecs = zeros(length(get_vector(model,"the")))
   for i in words[1:end]
      if i in vocabulary(model) && i ∉ stopwords(Languages.English())
         vecs = vecs .+ get_vector(model,i)
      end
   end
   return vecs
end

n = parse(Int64, get(parsed_args, "arg1", 0 ))
datatot = CSV.read("JonsData.csv", DataFrame)[(n-1)*1000+1:n*1000,:]
Random.seed!(n)
data, test = TrainTestSplit(datatot, .9);

vecs = zeros(size(data)[1],vecLength1 + vecLength2)
for i in 1:size(data)[1]
    vecs[i,:] = formulateText(M,data[i,1])
end

df = DataFrame(hcat(vecs,data[:,2]))

ii = vecLength1 + vecLength2
z=term(Symbol(:x, ii+1)) ~ sum(term.(Symbol.(names(df[:, Not(Symbol(:x, ii+1))]))))
logit = glm(z,df, Bernoulli(), LogitLink())

classtest = test[:,2]
vecstest = Matrix{Float64}(undef,length(classtest),ii)
for i in 1:length(classtest)
   vecstest[i,:] = formulateText(M,test[i,1])
end

       #Calculating Error Rate
artest = hcat(vecstest)

dftest = DataFrame(artest)

preds = GLM.predict(logit,dftest)

rez = preds.>.5

errors = 1 - sum(rez .== classtest)/length(rez)

save("Preds" * string(n) * ".jld", "val", preds)
save("Trues" * string(n) * ".jld", "val", classtest)
save("Errors" *string(n) * ".jld", "val", errors)
