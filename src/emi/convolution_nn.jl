using Flux
using LinearAlgebra
using Statistics
using ROCAnalysis
using MLBase
using Plots
using RecursiveArrayTools

include("../embeddings_nn.jl")
include("../data_cleaning.jl")
include("../PCA.jl")
include("../DTMCreation.jl")

field = " Cardiovascular / Pulmonary"
trainTestSplitPercent = .9

# Imports clean data
true_data = importClean()
sort!(true_data, "medical_specialty")

# Clean data and test/train split
# Creates both DTM (train & test)
sub_data = filtration(true_data, field)
train, test = TrainTestSplit(sub_data, trainTestSplitPercent)
dtm_train = CreateDTM(train, field)
dtm_test = CreateDTM(test, field)

# Finding classifcation vectors
class_train = dtm_train[end, :]
class_test = dtm_test[end, :]

# Removing classification columns
dtm_train = dtm_train[1:end-1, :]
dtm_test = dtm_test[1:end-1, :]

# Convolutional net
layers = Conv(tuple(100), 1 => 1, relu; bias=false)
nn(dtm_train[:,1])
nn = neural_net()
Flux.Conv(11911,100)
# Neural net architecture
function neural_net()
    nn = Chain(Conv(tuple(100), 1 => 1, relu; bias=false),
        Dense(15, 7, hardσ),
        Dense(7, 1, x->σ.(x))
        )
    return nn
end

# Makes DataLoader classes for test/train matrices
dl_test = Flux.Data.DataLoader((dtm_test, class_test'))
dl_train = Flux.Data.DataLoader((dtm_train, class_train'),
                                    batchsize = batchsize_custom, shuffle = true)
