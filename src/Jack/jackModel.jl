using Flux
using Lathe
using MLBase
using Plots
using Word2Vec

include("../data_cleaning.jl")
include("../embeddings_nn.jl")

vecLength1 = 15
window1 = 500

trainTestSplitPercent = .9
η = 0.05
epochs = 2000

paddingChoice = 1
batchsize_custom = 100


true_data = importClean()
sort!(true_data, "medical_specialty")
createCorpusText(true_data, paddingChoice)

# Medical field we're training on
field = " Cardiovascular / Pulmonary"

# Creating the embeddings using Word2Vec
word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true, window = window1)
M = Word2Vec.wordvectors("vectors.txt", normalize = false)
rm("vectors.txt")
rm("corpus.txt")

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

# Makes "DataLoader" classes for both testing and training data
# Batchsize for training shoulde be < ~size(train). Way less
newTestData = Flux.Data.DataLoader((test_mat, classTest'))
newTrainingData = Flux.Data.DataLoader((train_mat, class'),
                                    batchsize = batchsize_custom, shuffle = true)

nn = Chain(Dense(15, 7, relu),
            Dense(7, 4, relu),
            Dense(4, 1, x->σ.(x)))

opt = Flux.Optimiser(ExpDecay(), RADAM(0.001))

loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))

totalLoss = []
traceY = []
traceY2 = []
epochs = 1000
ps = Flux.params(nn)
    for i in 1:epochs
        Flux.train!(loss, ps, newTrainingData, opt)
        if i % 100 == 0
            println(i)
        end
        for (x,y) in newTrainingData
            totalLoss = loss(x,y)
        end
        push!(traceY, totalLoss)
        acc = 0
        for (x,y) in newTestData
            acc += sum((nn(x) .> .5) .== y)
        end
        decimalError = 1 - acc/length(classTest)
        percentError = decimalError * 100
        percentError = round(percentError, digits=2)
        push!(traceY2, percentError)
    end

# Testing for accuracy (at the end)
acc = 0
    for (x,y) in newTestData
        acc += sum((nn(x) .> .5) .== y)
    end

    decimalError = 1 - acc/length(classTest)
    percentError = decimalError * 100
    percentError = round(percentError, digits=2)
    print("Error Rate: ")
    print(percentError)
    println("%")

x = 1:epochs
y = traceY
plot(x, y, label = "Loss Progression")
xlabel!("Total # of epochs")
ylabel!("Loss values")

x2 = 1:epochs
y2 = traceY2
plot(x2, y2)
xlabel!("Total # of epochs")
ylabel!("Error Rate")

print(y2[argmin(y2)])
