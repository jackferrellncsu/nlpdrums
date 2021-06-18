using Flux
using LinearAlgebra
using Statistics
using ROCAnalysis
using MLBase
using Plots

include("../embeddings_nn.jl")
include("../data_cleaning.jl")
include("../PCA.jl")

# ---------------------------------------------------------------
# --------------------- Variables To Change ---------------------
# ---------------------------------------------------------------

# Vector length for embeddings, represents initial # of input nodes
# Window size (the smaller, the more syntactic; the larger, the more topical)
# Could add more than one set of embeddings, see "Word2VecReg.jl"
vecLength1 = 100
window1 = 50

trainTestSplitPercent = .9
η = 0.05
epochs = 2000

# 0 = no pad
# 1 = pad between field
# 2 = pad between docs
paddingChoice = 1
batchsize_custom = 1000
# Have to manually change the number of nodes in the nn layers
# in neural_network function

# ---------------------------------------------------------------
# ---------------- Calls from "embeddings_nn.jl" ----------------
# ---------------------------------------------------------------

# Imports clean data, creates "corpus.txt"
true_data = importClean()
sort!(true_data, "medical_specialty")
createCorpusText(true_data, paddingChoice)

# Medical field we're training on
field = " Cardiovascular / Pulmonary"

# Creating the embeddings using Word2Vec
word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true, window = window1)
M = wordvectors("vectors.txt", normalize = false)
rm("vectors.txt")
rm("corpus.txt")

# ---------------- Start Running Here For New Data Set ----------------

# Filtering new data and splitting train/test
# datasub = filtration(true_data, field)
data, test = TrainTestSplit(true_data, trainTestSplitPercent)

# Multinomial Field Sorting
fieldClass = []
uniqueFields = unique(true_data[:, 1])
for i in 1:length(data[:,1])
    push!(fieldClass, Flux.onehot(data[i, 1], uniqueFields))
end

testFieldClass = []
for i in 1:length(test[:,1])
    push!(testFieldClass, Flux.onehot(test[i, 1], uniqueFields))
end

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

field_mat = zeros(length(fieldClass), length(fieldClass[1]))
newTestMat = zeros(length(testFieldClass), length(testFieldClass[1]))

for (i, r) in enumerate(eachrow(field_mat))
    r .+= fieldClass[i]
end

for (i, r) in enumerate(eachrow(newTestMat))
    r .+= testFieldClass[i]
end

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
        Dense(100, 75, relu),
        Dense(75, 55, relu),
        Dense(55, 29, Flux.sigmoid)
        )
    return nn
end
# Makes "DataLoader" classes for both testing and training data
# Batchsize for training shoulde be < ~size(train). Way less
newTestData = Flux.Data.DataLoader((test_mat, newTestMat'))
newTrainingData = Flux.Data.DataLoader((train_mat, field_mat'),
                                    batchsize = batchsize_custom, shuffle = true)

# Defining our model, optimization algorithm and loss function
# @function Descent - gradient descent optimiser with learning rate η
nn = neural_net()
opt = ADAM()
loss(x, y) = sum(Flux.Losses.logitcrossentropy(nn(x), y))

# Actual training
totalLoss = []
traceY = []
traceY2 = []
ps = Flux.params(nn)
    for i in 1:epochs
        Flux.train!(loss, ps, newTrainingData, opt)
        println(i)
        for (x,y) in newTrainingData
            totalLoss = loss(x,y)
        end
        push!(traceY, totalLoss)
        acc = 0
        for (x,y) in newTestData
            acc += sum((classField(nn(x))) == vec(y'))
        end
        decimalError = 1 - acc/length(classTest)
        percentError = decimalError * 100
        percentError = round(percentError, digits=2)
        push!(traceY2, percentError)
    end

# Testing for accuracy (at the end)
acc = 0
    for (x,y) in newTestData
        acc += sum((classField(nn(x))) == vec(y'))
    end

# Printing the error rate from the accuracy function
decimalError = 1 - acc/length(classTest)
percentError = decimalError * 100
percentError = round(percentError, digits=2)
print("Error Rate: ")
print(percentError)
println("%")

# ---------------------------------------------------------------
# ------------------------ Visualization ------------------------
# ---------------------------------------------------------------

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

# Making an array for ROC curve plotting
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
    trueVec = Vector{Float64}(trueVec)

# Printing an ROC curve for word2vec
rocnums = MLBase.roc(classTest.==1, trueVec, 50)

emiTPR = true_positive_rate.(rocnums)
emiFPR = false_positive_rate.(rocnums)

plot(emiFPR,emiTPR)
plot!((0:100)./100, (0:100)./100, leg = false)
