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

using Flux
using LinearAlgebra
using Statistics
using ROCAnalysis
using MLBase
using Plots
using Word2Vec
using DataFrames
using GLM
using StatsBase
using CSV
using Languages
using Lathe.preprocess: TrainTestSplit
using LinearAlgebra
using JLD
using Random

function importClean()
    filename = "/Users/mlovig/Documents/GitHub/nlpdrums/src/cleanedData.csv"
    filepath = joinpath(@__DIR__, filename)
    arr = CSV.read(filename, DataFrame)

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

# ---------------------------------------------------------------
# --------------------- Variables To Change ---------------------
# ---------------------------------------------------------------

# ---------------- Start Running Here For New Data Set ----------------

# Filtering new data and splitting train/test
field = " Cardiovascular / Pulmonary"
n = parse(Int64, get(parsed_args, "arg1", 0 ))
n=10
Random.seed!(n)
D = importClean()
Random.seed!(n)
train, test = TrainTestSplit(D, .9);

Random.seed!(n)
createCorpusText(train,2)

#defining the lengths of the syntanctic or topical embeddings
vecLength1 = 25
vecLength2 = 25

#Defining the window sizes
window1 = 10
window2 = 300

#Creating the syntactic vector
word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true, window = window1, negative = 10, min_count = 0)
   M = wordvectors("vectors.txt", normalize = true)

   rm("vectors.txt")
word2vec("corpus.txt", "vectors.txt", size = vecLength2, verbose = true, window = window2, negative = 10, min_count = 0)
      M2 = wordvectors("vectors.txt", normalize = true)

      rm("vectors.txt")

   #creting the topical vectors

rm("corpus.txt")

vecsTrain = zeros(size(data)[1],vecLength1 + vecLength2)
for i in 1:size(data)[1]
         vecsTrain[i,:] = vcat(formulateText(M,data[i,3]),formulateText(M2,data[i,3]))
end
vecsTest = zeros(size(test)[1],vecLength1 + vecLength2)
for i in 1:size(test)[1]
         vecsTest[i,:] = vcat(formulateText(M,test[i,3]),formulateText(M2,test[i,3]))
end
class = data[:,2] .== field
classTest = test[:,2] .== field
# creating the matrix to run through nn
train_mat = vecsTrain'
test_mat = vecsTest'

# ---------------------------------------------------------------
# --------------------- Neural Net Training ---------------------
# ---------------------------------------------------------------

# creation of neural network architecture
# @function Dense - takes in input, output, and activation
# function; creates dense layer based on parameters.
# @return nn - both dense layers tied together
function neural_net()
    nn = Chain(
        Dense(15, 30, swish),Dense(30, 10, swish),
        Dense(10, 1, x->σ.(x))
        )
    return nn
end

# Makes "DataLoader" classes for both testing and training data
# Batchsize for training shoulde be < ~size(train). Way less
newTestData = Flux.Data.DataLoader((test_mat, classTest'))
newTrainingData = Flux.Data.DataLoader((train_mat, class'), shuffle = true)

# Defining our model, optimization algorithm and loss function
# @function Descent - gradient descent optimiser with learning rate η
nn = neural_net()
opt = RADAM()
ps = Flux.params(nn)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))

# Actual training
traceY = []
traceY2 = []
epochs = 500
    for i in 1:epochs
        Flux.train!(loss, ps, newTrainingData, opt)
        if i % 100 == 0
            println(i)
        end
        #=
        for (x,y) in trainDL
            totalLoss = loss(x,y)
        end
        push!(traceY, totalLoss)
        =#
        # acc = 0
        # for (x,y) in testDL
        #     acc += sum((nn(x) .> .5) .== y)
        # end
        # decimalError = 1 - acc/length(classVal)
        # percentError = decimalError * 100
        # percentError = round(percentError, digits=2)
        # push!(traceY2, percentError)
    end

acc = 0
preds = []
trues = []
    for (x,y) in newTestData
        push!(preds, nn(x)[1])
        push!(trues,y[1])
    end

errors = 1 - sum((preds .> .5) .== trues)/length(classTest)

JLD.save("Preds" * string(n) * ".jld", "val", preds)
JLD.save("Trues" * string(n) * ".jld", "val", classtest)
JLD.save("Errors" *string(n) * ".jld", "val", errors)

# ---------------------------------------------------------------
# ------------------------ Visualization ------------------------
# --------------------------------------------------------------
