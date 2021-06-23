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
batchsize_custom = 100
η = 0.01
epochs = 1000


# Imports clean data
true_data = importClean()
sort!(true_data, "medical_specialty")

# Clean data and test/train split
# Creates DTM (train & test)
DTM = CreateDTM(true_data, field)
total_DTM = DataFrame(DTM')
train, test = TrainTestSplit(total_DTM, trainTestSplitPercent)

# Finding classifcation vectors
class_train = train[:, end]
class_test = test[:, end]

# Removing classification columns
dtm_train = convert(Matrix, train[:, 1:end-1])
dtm_test = convert(Matrix, test[:, 1:end-1])

# Convolution & Pooling for training matrix
train1 = length(dtm_train[:,1])
train2 = length(dtm_train[1,:])
train_array = reshape(dtm_train, (train1, train2, 1, 1))
layers = Chain(
        Conv(tuple(1, 12), 1 => 1, relu),
        # VocabSize/PoolingSize = # of predictors
        MaxPool(tuple(1, 100)))
conv_train_array = layers(train_array)
conv_train_mat = conv_train_array[1, :]'
for i in 2:length(conv_train_array[:,1])
    println(i)
    conv_train_mat = vcat(conv_train_mat, conv_train_array[i,:]')
end

# Convolution & Pooling for testing matrix
test1 = length(dtm_test[:,1])
test2 = length(dtm_test[1,:])
test_array = reshape(dtm_test, (test1, test2, 1, 1))
layer = Chain(
        Conv(tuple(11, 1), 1 => 1, relu),
        # Must be opposite to pooling layer above
        MaxPool(tuple(100, 1)))
conv_test_array = layers(test_array)
conv_test_mat = conv_test_array[1, :]'
for i in 2:length(conv_test_array[:,1])
    println(i)
    conv_test_mat = vcat(conv_test_mat, conv_test_array[i,:]')
end

# Neural net architecture
function neural_net()
    nn = Chain(
        Dense(227, 150, relu),
        Dense(150, 75, relu),
        Dense(75, 40, relu),
        Dense(40, 15, relu),
        Dense(15, 7, relu),
        Dense(7, 1, x->σ.(x))
        )
    return nn
end

# Makes DataLoader classes for test/train matrices
dl_test = Flux.Data.DataLoader((conv_test_mat', class_test'))
dl_train = Flux.Data.DataLoader((conv_train_mat', class_train'),
                                    batchsize = batchsize_custom, shuffle = true)

nn = neural_net()
opt = Descent(η)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))

# Actual training
totalLoss = []
traceY = []
traceY2 = []
ps = Flux.params(nn)
    for i in 1:epochs
        Flux.train!(loss, ps, dl_train, opt)
        println(i)
        for (x,y) in dl_train
            totalLoss = loss(x,y)
        end
        push!(traceY, totalLoss)
        acc = 0
        for (x,y) in dl_test
            acc += sum((nn(x) .> .5) .== y)
        end
        decimalError = 1 - acc/length(class_test)
        percentError = decimalError * 100
        percentError = round(percentError, digits=2)
        push!(traceY2, percentError)
    end

# Testing for accuracy (at the end)
acc = 0
    for (x,y) in dl_test
        acc += sum((nn(x) .> .5) .== y)
    end

# Printing the error rate from the accuracy function
decimalError = 1 - acc/length(class_test)
percentError = decimalError * 100
percentError = round(percentError, digits=2)
print("Error Rate: ")
print(percentError)
println("%")

# --------------------- Plotting ---------------------

# Trace Plot 1 (loss vs epochs)
x = 1:epochs
y = traceY
plot(x, y, label = "Loss Progression")
xlabel!("Total # of epochs")
ylabel!("Loss values")

# Trace Plot 2 (test set accuracy vs epochs)
x = 1:epochs
y = traceY2
plot(x, y)
xlabel!("Total # of epochs")
ylabel!("Error Rate")
