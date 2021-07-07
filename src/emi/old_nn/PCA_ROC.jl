using Flux
using LinearAlgebra
using Statistics
using ROCAnalysis
using MLBase
using Plots

include("../DTMCreation.jl")
include("../embeddings_nn.jl")
include("../data_cleaning.jl")
include("../PCA.jl")

# ---------------------------------------------------------------
# --------------------- Variables To Change ---------------------
# ---------------------------------------------------------------

# Don't forget to change neural net layer values
trainTestSplitPercent = .9
η = 0.05
epochs = 1000
NumPC = 27

# Have to manually change the number of nodes in the nn layers
# in neural_network function

# ---------------- Start Running Here For New Data Set ----------------

importedData = importClean()
dtmi = CreateDTM(importedData, " Cardiovascular / Pulmonary")
df = DataFrame(1.0*dtmi', :auto)

# Filtering new data and splitting train/test
data, test = TrainTestSplit(df, trainTestSplitPercent)

# PCA/SVD matrices
Us, sigs, Vts = PCAVecs(Matrix(data)[:, 1:end - 1], 50)
U = Us[NumPC]
sig = sigs[NumPC]
Vt = Vts[NumPC]

UsTest, sigsTest, VtsTest = PCAVecs(Matrix(test)[:, 1:end - 1], 50)
UTest = UsTest[NumPC]
STest = sigsTest[NumPC]
VTest = VtsTest[NumPC]

# Making the classification column for after splitting
field = " Cardiovascular / Pulmonary"

class = data[:,end]
classTest = test[:,end]

tmat = Matrix(test)

# creating the matrix to run through nn
train_mat = U'
test_mat = UTest'

# ---------------------------------------------------------------
# --------------------- Neural Net Training ---------------------
# ---------------------------------------------------------------

# creation of neural network architecture
# @function Dense - takes in input, output, and activation
# function; creates dense layer based on parameters.
# @return nn - both dense layers tied together
function neural_net()
    nnPCA = Chain(
            Dense(27, 15, hardσ),
            Dense(15, 1, x->σ.(x))
            )
    return nnPCA
end

# Makes "DataLoader" classes for both testing and training data
# Batchsize for training shoulde be < ~size(train). Way less
newTestData = Flux.Data.DataLoader((test_mat, classTest'))
newTrainingData = Flux.Data.DataLoader((train_mat, class'),
                                    batchsize = 100, shuffle = true)

# Defining our model, optimization algorithm and loss function
# @function Descent - gradient descent optimiser with learning rate η
nnPCA = neural_net()
opt = Descent(η)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(nnPCA(x), y))

# Actual training of neural net
ps = Flux.params(nnPCA)

for i in 1:epochs
        Flux.train!(loss, ps, newTrainingData, opt)
    end

# Testing for accuracy
acc = 0
    for (x,y) in newTestData
        acc += sum((nnPCA(x) .> .5) .== y)
    end

# Printing the error rate from the accuracy function
decimalError = 1 - acc/length(classTest)
percentError = decimalError * 100
percentError = round(percentError, digits=2)
print("Error Rate: ")
print(percentError)
println("%")

using BSON: @save
@save "src/emi/nnPCA.bson" nnPCA

# Making an array for ROC curve plotting
realVec = []
    for (x, y) in newTestData
        push!(realVec, nnPCA(x))
    end
j = []
trueVec = []
    for x in realVec
        for i in x
            push!(trueVec, i)
        end
    end
trueVec = Vector{Float64}(trueVec)

# Printing an ROC curve for PCA
rocnums = MLBase.roc(classTest.==1, trueVec, 50)

emiTPR = true_positive_rate.(rocnums)
emiFPR = false_positive_rate.(rocnums)

plot(emiFPR,emiTPR)
plot!((0:100)./100, (0:100)./100, leg = false)
