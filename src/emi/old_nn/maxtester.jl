using Flux
using LinearAlgebra
using Statistics
using ROCAnalysis
using MLBase
using Plots
using Plotly
using CSV

include("../embeddings_nn.jl")
include("../emi/embeddingsForMax.jl")
include("../data_cleaning.jl")
include("../PCA.jl")

# ---------------------------------------------------------------
# --------------------- Variables To Change ---------------------
# ---------------------------------------------------------------

# Vector length for embeddings, represents initial # of input nodes
# Window size (the smaller, the more syntactic; the larger, the more topical)
# Could add more than one set of embeddings, see "Word2VecReg.jl"
vecLength1 = 15
window1 = 3

trainTestSplitPercent = .9
η = 0.05
epochs = 250

# 0 = no pad
# 1 = pad between field
# 2 = pad between docs
paddingChoice = 0
batchsize_custom = 100
# Have to manually change the number of nodes in the nn layers
# in neural_network function

# ---------------------------------------------------------------
# ---------------- Calls from "embeddings_nn.jl" ----------------
# ---------------------------------------------------------------

true_data = CSV.read("wordy.csv", DataFrame)
# Imports clean data, creates "corpus.txt"
createCorpusText(true_data[:,1], paddingChoice)

# Creating the embeddings using Word2Vec
word2vec("English2.txt", "vectors.txt", size = vecLength1, verbose = true, window = window1)
M = wordvectors("vectors.txt", normalize = false)
rm("vectors.txt")


# ---------------- Start Running Here For New Data Set ----------------

# Filtering new data and splitting train/test
# datasub = filtration(true_data, field)
data, test = TrainTestSplit(true_data, trainTestSplitPercent)

# Creating classification columns
class = data[:,2]
class = class * 1.0

classTest = test[:, 2]
classTest = classTest * 1.0

tmat = Matrix(test)

# Creating an empty matrix for training where the length size is
# double the medical field chosen and the width is the sum of two
# vectors lengths (or one vector if using one)
vecsTrain = zeros(length(class),vecLength1)
vecsTest = zeros(length(tmat[:, 1]), vecLength1)

# Places embeddings in empty matrix
for i in 1:length(class)
    vecsTrain[i,:] = formulateText(M,data[i,1])
end

for i in 1:length(tmat[:, 1])
    vecsTest[i,:] = formulateText(M, test[i,1])
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
        Dense(15, 7, relu),
        Dense(7, 1, Flux.sigmoid)
        )
    return nn
end
# Makes "DataLoader" classes for both testing and training data
# Batchsize for training shoulde be < ~size(train). Way less
newTestData = Flux.Data.DataLoader((test_mat, classTest'))
newTrainingData = Flux.Data.DataLoader((train_mat, class'),
                                    batchsize = batchsize_custom, shuffle = true)

# Defining our model, optimization algorithm and loss function
# @function Descent - gradient descent optimiser with learning rate η
nn = neural_net()
opt = Flux.Optimiser(ExpDecay(0.01, 0.1, 50, 1e-4), RADAM())
loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))
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
        acc += (nn(x) .> .5) == y
    end
    decimalError = 1 - acc/length(classTest)
    percentError = decimalError * 100
    percentError = round(percentError, digits=2)
    push!(traceY2, percentError)
end


# Testing for accuracy (at the end)
acc = 0
    for (x,y) in newTestData
        acc += (nn(x) .> .5) == y
    end

# Printing the error rate from the accuracy function
decimalError = 1 - acc/length(classTest)
percentError = decimalError * 100
percentError = round(percentError, digits=2)
print("Error Rate: ")
print(percentError)
println("%")

# ---------------------------------------------------------------
# ----------------------- Conformal Ideas -----------------------
# ---------------------------------------------------------------
#=
conf_mat = train_mat
class_mat = diagm(ones(29))
new_class_mat = field_mat

for i in 1:length(class_mat[1,:])
    new_class_mat = vcat(new_class_mat, class_mat[i, :]')
    conf_mat = vcat(conf_mat, test_mat[i, :]')
end
=#
# ---------------------------------------------------------------
# ------------------------ Visualization ------------------------
# ---------------------------------------------------------------

# Trace Plot 1 (loss vs epochs)
x = 1:epochs
y = traceY
plot(x, y, label = "Loss Progression")
xlabel!("Total # of epochs")
ylabel!("Loss values")

#-------------------------------------------------------------------------------
# Trace Plot 2 (test set accuracy vs epochs)
x = 1:epochs
y = traceY2
plot(x, y)
xlabel!("Total # of epochs")
ylabel!("Error Rate")

#-------------------------------------------------------------------------------
# Bar Chart (Accuracy by field)
allFields = []
allFields = vec(unique!(true_data[:,1]))
rlyAllFields = []
for i in 1:length(allFields)
    push!(rlyAllFields, allFields[i])
end
correct_result = Vector{String}()
incorrect_result = Vector{String}()
incorrect_preds = Vector{String}()
for (x,y) in newTestData
    if classField(nn(x)) == vec(y')
        push!(correct_result, uniqueFields[argmax(y)])
    else
        push!(incorrect_preds, uniqueFields[argmax(nn(x))])
        push!(incorrect_result, uniqueFields[argmax(y)])
    end
end
c1 = countmap(correct_result)
c3 = countmap(incorrect_preds)
c2 = countmap(incorrect_result)
correctplot = []
incorrectplot = []
predplot = []
for i in uniqueFields
    push!(correctplot,get(c1, i, 0))
    push!(incorrectplot,get(c2, i, 0))
    push!(predplot, get(c3, i, 0))
end
totplot = correctplot .+ incorrectplot
correctplot1 = correctplot ./ totplot
incorrectplot1 = incorrectplot ./ totplot
# Proportional bar chart
groupedbar(rlyAllFields, [correctplot1 incorrectplot1], xrotation = 45,
                bar_position = :stack, label = ["Correct" "Incorrect"],
                xlabel = "Medical Fields", ylabel = "Proportions")
# Not proportional bar chart
groupedbar(rlyAllFields, [correctplot incorrectplot], xrotation = 45,
                bar_position = :stack, label = ["Correct" "Incorrect"],
                xlabel = "Medical Fields", ylabel = "Transcript Count")
#
groupedbar(rlyAllFields, [correctplot predplot], xrotation = 45,
                bar_position = :stack, label = ["correct pred" "incorrect pred"],
                xlabel = "Medical Fields", ylabel = "Transcript Count")


#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
# Printing an ROC curve for word2vec
rocnums = MLBase.roc(classTest.==1, trueVec, 50)

emiTPR = true_positive_rate.(rocnums)
emiFPR = false_positive_rate.(rocnums)

plot(emiFPR,emiTPR)
plot!((0:100)./100, (0:100)./100, leg = false)
