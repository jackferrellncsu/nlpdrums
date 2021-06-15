using Flux
using LinearAlgebra
using Statistics
using ROCAnalysis
using MLBase

include("../embeddings_nn.jl")
include("../data_cleaning.jl")
include("../PCA.jl")

# ---------------------------------------------------------------
# --------------------- Variables To Change ---------------------
# ---------------------------------------------------------------

# Vector length for embeddings (Max 50), represents initial # of input nodes
# Window size (the smaller, the more syntactic; the larger, the more topical)
# Could add more than one set of embeddings, see "Word2VecReg.jl"
vecLength1 = 15
window1 = 15

trainTestSplitPercent = .9
η = 0.05
epochs = 1000

# Have to manually change the number of nodes in the nn layers
# in neural_network function

# ---------------------------------------------------------------
# ---------------- Calls from "embeddings_nn.jl" ----------------
# ---------------------------------------------------------------

# Imports clean data, creates "corpus.txt"
true_data = importClean()
sort!(true_data, "medical_specialty")
createCorpusText(true_data, 0)

# Medical field we're training on
field = " Cardiovascular / Pulmonary"

# Creating the embeddings using Word2Vec
word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true, window = window1)
M = wordvectors("vectors.txt", normalize = false)
rm("vectors.txt")
rm("corpus.txt")

# ---------------- Start Running Here For New Data Set ----------------

# Filtering new data and splitting train/test
datasub = filtration(true_data, field)
data, test = TrainTestSplit(datasub, trainTestSplitPercent)


# Making the classification column for after splitting
class = data[:,1] .== field
class = class * 1.0

classTest = test[:, 1] .== field
classTest = classTest * 1.0

tmat = Matrix(test)

# Creating an empty matrix for training where the length size is
# double the medical field chosen and the width is the sum of two
# vectors lengths (or one vector if using one)
vecsTrain = zeros(length(class),vecLength1)
vecsTest = zeros(length(tmat[:, 1]), vecLength1)

# Places embeddings in empty matrix
for i in 1:length(class)
    vecsTrain[i,:] = formulateText(M,data[i,3])
end

for i in 1:length(tmat[:, 1])
    vecsTest[i,:] = formulateText(M, test[i,3])
end

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
            Dense(15, 7, hardσ),
            Dense(7, 1, x->σ.(x))
            )
    return nn
end

# Makes "DataLoader" classes for both testing and training data
# Batchsize for training shoulde be < ~size(train). Way less
newTestData = Flux.Data.DataLoader((test_mat, classTest'))
newTrainingData = Flux.Data.DataLoader((train_mat, class'),
                                    batchsize = 40, shuffle = true)

# Defining our model, optimization algorithm and loss function
# @function Descent - gradient descent optimiser with learning rate η
nn = neural_net()
opt = Descent(η)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))

# Actual training
ps = Flux.params(nn)
    for i in 1:epochs
        Flux.train!(loss, ps, newTrainingData, opt)
    end

# Testing for accuracy
acc = 0
    for (x,y) in newTestData
        acc += sum((nn(x) .> .5) .== y)
    end

realVec = []
    for (x, y) in newTestData
        push!(realVec, nn(x))
    end
j = []
trueVec = []
    for x in realVec
        for i in x
            push!(trueVec, i)
        end
    end




# Printing the error rate from the accuracy function
decimalError = 1 - acc/length(classTest)
percentError = decimalError * 100
percentError = round(percentError, digits=2)
print("Error Rate: ")
print(percentError)
println("%")

trueVec = Vector{Float64}(trueVec)
