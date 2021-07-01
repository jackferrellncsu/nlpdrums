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
using Embeddings

function importClean()
    filename = "/Users/mlovig/Documents/GitHub/nlpdrums/src/cleanedData.csv"
    filepath = joinpath(@__DIR__, filename)
    arr = CSV.read(filename, DataFrame)

    return arr
end

function get_embedding(word)
    ind = get_word_index[word]
    emb = embtable.embeddings[:,ind]
    return emb
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

function formulateText(script)
   words = split(script, " ")
   vecs = zeros(300)
   for i in words[1:end]
       if get(get_word_index, i , -1) != -1
         vecs = vecs .+ get_embedding(i)
     end
   end
   return vecs
end

embtable = load_embeddings(FastText_Text, max_vocab_size = 50000)
JLD.save("embtable.jld", "embtable", embtable)

t = JLD.load("embtable.jld")
embtable = t["embtable"]
get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

# ---------------------------------------------------------------
# --------------------- Variables To Change ---------------------
# ---------------------------------------------------------------

# ---------------- Start Running Here For New Data Set ----------------

# Filtering new data and splitting train/test

n = parse(Int64, get(parsed_args, "arg1", 0 ))

p = [20,40,60,80,100,120,140,160,180,200]
seed = n % 100
par = Int(ceil(n/100))
param = 200

datatot = CSV.read("src/cleanedData.csv", DataFrame)
Random.seed!(13)
datatot = filtration(datatot, " Cardiovascular / Pulmonary")
datame = DataFrame(hcat(datatot[:,3], (datatot[:,1].== " Cardiovascular / Pulmonary")))

Random.seed!(13)
data, val = TrainTestSplit(datame, .7)

    #defining the lengths of the syntanctic or topical embeddings

vecsTrain = zeros(size(data)[1],300)
for i in 1:size(data)[1]
             vecsTrain[i,:] = formulateText(data[i,1])
end

vecsVal = zeros(size(val)[1],300)
for i in 1:size(val)[1]
             vecsVal[i,:] = formulateText(val[i,1])
end
class = data[:,2] .== 1
classTest = val[:,2] .== 1
    # creating the matrix to run through nn
train_mat = vecsTrain'
test_mat = vecsVal'

    # ---------------------------------------------------------------
    # --------------------- Neural Net Training ---------------------
    # ---------------------------------------------------------------

    # creation of neural network architecture
    # @function Dense - takes in input, output, and activation
    # function; creates dense layer based on parameters.
    # @return nn - both dense layers tied together
function neural_net()
        nn = Chain(
            Dense(300, param, swish),Dense(param, 20, swish),Dense(20, 1, x->σ.(x))
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

epochs = 500
    for i in 1:epochs
            Flux.train!(loss, ps, newTrainingData, opt)
            println(i)
    end

acc = 0
preds = []
truez = []
for (x,y) in newTestData
            push!(preds, nn(x)[1])
            push!(truez,y[1])
end

errors = 1 - sum((preds .> .5) .== truez)/length(classTest)

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

Plots.title!("Fast Text Neural Net")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")
xaxis!([0,1])
yaxis!([0,1])

JLD.save("Errors" *string(par) * "_" * string(seed)* ".jld", "val", errors)

# ---------------------------------------------------------------
# ------------------------ Visualization ------------------------
# --------------------------------------------------------------
