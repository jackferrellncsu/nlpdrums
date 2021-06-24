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

# Vector length for embeddings (Max 50), represents initial # of input nodes
# Window size (the smaller, the more syntactic; the larger, the more topical)
# Could add more than one set of embeddings, see "Word2VecReg.jl"
M = wordvectors("vectors1.txt", normalize = false)

M2 = wordvectors("vectors2.txt", normalize = false)
# Have to manually change the number of nodes in the nn layers
# in neural_network function


# ---------------- Start Running Here For New Data Set ----------------

# Filtering new data and splitting train/test
n = parse(Int64, get(parsed_args, "arg1", 0 ))
datatot = CSV.read("JonsData.csv", DataFrame)[(n-1)*1000+1:n*1000,:]
Random.seed!(n)
data, test = TrainTestSplit(datatot, .9);

vecsTrain = zeros(size(data)[1],vecLength1 + vecLength2)
for i in 1:size(data)[1]
         vecsTrain[i,:] = vcat(formulateText(M,data[i,1]),formulateText(M2,data[i,1]))
end
vecsTest = zeros(size(test)[1],vecLength1 + vecLength2)
for i in 1:size(test)[1]
         vecsTest[i,:] = vcat(formulateText(M,test[i,1]),formulateText(M2,test[i,1]))
end
class = data[:,2]
classTest = test[:,2]
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

save("Preds" * string(n) * ".jld", "val", preds)
save("Trues" * string(n) * ".jld", "val", classtest)
save("Errors" *string(n) * ".jld", "val", errors)

# ---------------------------------------------------------------
# ------------------------ Visualization ------------------------
# --------------------------------------------------------------
