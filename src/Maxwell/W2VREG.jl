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

function createCorpusText(data, choice)
   allDocs = ""
   thePad = ""
   for i in 1:3000
      thePad = thePad * " randomWordNow"
   end
   for i in 1:length(data[:, 3])
      println(i)
      if choice == 1
         if i != 1
            if data[i, 1] != data[i-1, 1]
               println("This is a seperation")
               allDocs = allDocs * thePad * " " * data[i, 3]
            else
               allDocs = allDocs * " " * data[i, 3]
            end
         end
      elseif choice == 0
         allDocs = allDocs * " " * data[i, 3]
      elseif choice == 2
         allDocs = allDocs * thePad * " " * data[i, 3]
      end
   end
   open("corpus.txt","a") do io
      println(io,allDocs)
   end
end

function importClean()
    filename = "cleanedData.csv"
    filepath = joinpath(@__DIR__, filename)
    arr = CSV.read(filepath, DataFrame)

    return arr
end

function filtration(df, field)
   indexes = []
   for i in 1:length(df[:,1])
      if df[i,1] == field
         push!(indexes,i)
      else
         if rand() < sum(df[:,1].==field)/(length(df[:,1]) - sum(df[:,1].==field))
            push!(indexes,i)
         end
      end
   end
   return df[indexes,:]
end

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

field = " Cardiovascular / Pulmonary"
n = parse(Int64, get(parsed_args, "arg1", 0 ))
Random.seed!(n)
D = CSV.read("/Users/mlovig/Documents/GitHub/nlpdrums/src/cleanedData.csv", DataFrame)
Random.seed!(n)
train, test = TrainTestSplit(D, .9);

Random.seed!(n)
data = filtration(train, " Cardiovascular / Pulmonary")
createCorpusText(train,0)

#defining the lengths of the syntanctic or topical embeddings
vecLength1 = 15

#Defining the window sizes
window1 = 50

#Creating the syntactic vector
word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true, window = window1, negative = 10, min_count = 0)
   M = wordvectors("vectors.txt", normalize = true)

   rm("vectors.txt")
   #creting the topical vectors

rm("corpus.txt")

vecs = zeros(size(data)[1],vecLength1)
for i in 1:size(data)[1]
    vecs[i,:] = formulateText(M,data[i,1])
end

df = DataFrame(hcat(vecs,data[:,2]))

ii = vecLength1
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

JLD.save("Preds" * string(n) * ".jld", "val", preds)
JLD.save("Trues" * string(n) * ".jld", "val", classtest)
JLD.save("Errors" *string(n) * ".jld", "val", errors)
