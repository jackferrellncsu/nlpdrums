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
# ---------------- Calls from "embeddings_nn.jl" ----------------
# ---------------------------------------------------------------

# Imports clean data, creates "corpus.txt"
true_data = importClean()
sort!(true_data, "medical_specialty")
createCorpusText(true_data, 1)

# Medical field we're training on
field = " Cardiovascular / Pulmonary"

# ---------------------------------------------------------------
# -------------------- vecLength for loop -----------------------
# ---------------------------------------------------------------

vecLength = 15
window1 = 500
trainTestSplitPercent = .9
η = 0.05
epochs = 100
traceY3 = []

for i in 1:vecLength
    println(i, i, i, i, i)

    # Creating the embeddings using Word2Vec
    word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true, window = window1)
    M = wordvectors("vectors.txt", normalize = false)
    rm("vectors.txt")

    # Filtering new data and splitting train/test
    datasub = filtration(true_data, field)
    data, test = TrainTestSplit(datasub, trainTestSplitPercent)


    # Making the classification column for after splitting
    class = data[:,1] .== field
    class = class * 1.0

    classTest = test[:, 1] .== field
    classTest = classTest * 1.0

    tmat = Matrix(test)

    vecsTrain = zeros(length(class),vecLength1)
    vecsTest = zeros(length(tmat[:, 1]), vecLength1)

    for i in 1:length(class)
        vecsTrain[i,:] = formulateText(M,data[i,3])
    end

    for i in 1:length(tmat[:, 1])
        vecsTest[i,:] = formulateText(M, test[i,3])
    end

    train_mat = vecsTrain'
    test_mat = vecsTest'

# ---------------------------------------------------------------
# --------------------- Neural Net Training ---------------------
# ---------------------------------------------------------------

    function neural_net()
        nn = Chain(
            Dense(30, 21, hardσ),
            Dense(21, 15, hardσ),
            Dense(15, 7, hardσ),
            Dense(7, 3, hardσ),
            Dense(3, 1, x->σ.(x))
            )
        return nn
    end

    # Makes "DataLoader" classes for both testing and training data
    # Batchsize for training shoulde be < ~size(train). Way less
    newTestData = Flux.Data.DataLoader((test_mat, classTest'))
    newTrainingData = Flux.Data.DataLoader((train_mat, class'),
                                    batchsize = 100, shuffle = true)

    # Defining our model, optimization algorithm and loss function
    # @function Descent - gradient descent optimiser with learning rate η
    nn = neural_net()
    opt = Descent(η)
    loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))

    # Actual training
    ps = Flux.params(nn)
    for i in 1:epochs
        Flux.train!(loss, ps, newTrainingData, opt)
        println(i)
    end

    # Accuracy
    for (x,y) in newTestData
        acc += sum((nn(x) .> .5) .== y)
    end
    decimalError = 1 - acc/length(classTest)
    percentError = decimalError * 100
    percentError = round(percentError, digits=2)
    push!(traceY3, percentError)

end
# ---------------------------------------------------------------
# ------------------------ Visualization ------------------------
# ---------------------------------------------------------------

x = 1:vecLength
y = traceY3
plot(x, y)
xlabel!("Total vecLength")
ylabel!("Error Rate")
