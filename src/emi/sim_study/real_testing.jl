using Flux
using LinearAlgebra
using Statistics
using ROCAnalysis
using MLBase
using Plots
using RecursiveArrayTools
using TextAnalysis
using InvertedIndices
using JLD
using StatsBase
using Random
using CSV
using Lathe.preprocess: TrainTestSplit

include("../PCA.jl")
include("../data_cleaning.jl")
include("../embeddings_nn.jl")
include("../DTMCreation.jl")

# ====================== Variables ====================== #

trainTestSplitPercent = .7
batchsize_custom = 100
epochs = 500

errorrates = []
predictions = []
trueValues = []

# ====================== Conv Loop ===================== #

true_data = importClean()
sort!(true_data, "medical_specialty")

field = " Cardiovascular / Pulmonary"
Random.seed!(13)
DTM = CreateDTM(true_data, field)
total_DTM = DataFrame(DTM')

train, test = TrainTestSplit(total_DTM, trainTestSplitPercent)

println("DTM Done")

# Finding classifcation vectors
class_train = train[:, end]
class_test = test[:, end]

# Removing classification columns
dtm_train = Matrix(train[:, 1:end-1])
dtm_test = Matrix(test[:, 1:end-1])

########################## Beginning of Convolution ##########################

# Convolutional Layer
num_rows_train = length(dtm_train[:,1])
num_rows_test = length(dtm_test[:,1])
layers_train = Chain(
                Conv(tuple(1, 20), 1 => 1, relu),
                AdaptiveMaxPool(tuple(num_rows_train, 600)))
layers_test = Chain(
                Conv(tuple(1, 20), 1 => 1, relu),
                AdaptiveMaxPool(tuple(num_rows_test, 600)))
println("Start Conv")

# Convolution & Pooling for training matrix
train1 = length(dtm_train[:,1])
train2 = length(dtm_train[1,:])
train_array = reshape(dtm_train, (train1, train2, 1, 1))
conv_train_array = layers_train(train_array)
conv_train_mat = conv_train_array[1, :]'
for i in 2:length(conv_train_array[:,1])
    global conv_train_mat = vcat(conv_train_mat, conv_train_array[i,:]')
end

# Convolution & Pooling for testing matrix
test1 = length(dtm_test[:,1])
test2 = length(dtm_test[1,:])
test_array = reshape(dtm_test, (test1, test2, 1, 1))
conv_test_array = layers_test(test_array)
conv_test_mat = conv_test_array[1, :]'
for i in 2:length(conv_test_array[:,1])
    global conv_test_mat = vcat(conv_test_mat, conv_test_array[i,:]')
end

# Making layers for neural net
L1 = length(conv_test_mat[1,:]) #600
L2 = Int(ceil(L1/3)) # 200
L3 = Int(ceil(L2/3)) # 67
L4 = Int(ceil(L3/3)) # 23
L5 = Int(ceil(L4/3)) # 8

# Neural net architecture
function neural_net()
    nn = Chain(
        Dense(L1, L2, relu),
        Dense(L2, L3, relu),
        Dense(L3, L4, relu),
        Dense(L4, L5, relu),
        Dense(L5, 1, x->σ.(x))
        )
    return nn
end

# Makes DataLoader classes for test/train matrices
dl_test = Flux.Data.DataLoader((conv_test_mat', class_test'))
dl_train = Flux.Data.DataLoader((conv_train_mat', class_train'),
                                    batchsize = batchsize_custom, shuffle = true)

nn = neural_net()
opt = RADAM()
loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))
println("Start Train")

# Actual training
ps = Flux.params(nn)
    for i in 1:epochs
        println(string(i))
        Flux.train!(loss, ps, dl_train, opt)
    end

# Testing for accuracy (at the end)
temppreds = []
for (x,y) in dl_test
    push!(temppreds,nn(x)[1])
end

push!(errorrates, 1-(sum((temppreds .> .5) .== class_test)/size(class_test)[1]))

println("Round ",length(errorrates) , ": ", round((errorrates[end] * 100), digits = 2), "%")
