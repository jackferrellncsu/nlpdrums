using Flux
using LinearAlgebra
using Statistics
using ROCAnalysis
using MLBase
using Plots
using Plotly
using CSV
using Random

Random.seed!(5)
rand()

include("../embeddings_nn.jl")
include("../emi/embeddingsForMax.jl")
include("../data_cleaning.jl")
include("../PCA.jl")
include("../jld2results.jl")

# veclength = # of input nodes
veclength = 15

# windowsize: smaller = more syntactical, larger = more topical
windowsize = 5

TrainTestSplitPercent = 0.9
lr = 0.05
epochs = 500

# padding options:
# 0 = no padding
# 1 = padding between medical specialties
# 2 = padding between every doc
paddingchoice = 0
batchsize_ = 100

data = CSV.read("JonsData.csv", DataFrame)
#createCorpusText(data[:,1], paddingchoice)

# creating embeddings
word2vec("corpus.txt", "vectors.txt", size = veclength, verbose = true,
                                                            window = windowsize)
M = wordvectors("vectors.txt", normalize = false)
rm("vectors.txt")

errors = []
predictions = []
trueValues = []

for i in 1:1000
    Random.seed!(i)
    #======== Run From Here for new data =======#

    train, test = TrainTestSplit(data[(i-1)*1000+1:(i)*1000, :], TrainTestSplitPercent)

    classTrain = train[:,2]
    classTrain = classTrain * 1.0

    classTest = test[:,2]
    classTest = classTest * 1.0

    tmat = Matrix(test)


    vecsTrain = zeros(length(classTrain), veclength)
    vecsTest = zeros(length(tmat[:,1]), veclength)

    for i in 1:length(classTrain)
        vecsTrain[i,:] = formulateText(M, train[i,1])
    end

    for i in 1:length(tmat[:,1])
        vecsTest[i,:] = formulateText(M, test[i,1])
    end

    train_mat = vecsTrain'
    test_mat = vecsTest'

    # =============Neural Net Stuff=============#

    function neural_net()
        nn = Chain(
            Dense(15, 7, mish),
            Dropout(0.6),
            Dense(7, 1, x->σ.(x))
        )
    end

    newTrainData = Flux.Data.DataLoader((train_mat, classTrain'), batchsize = batchsize_,
                                                                    shuffle = true)
    newTestData = Flux.Data.DataLoader((test_mat, classTest'))


    neuralnet = neural_net()
    Flux.testmode!(neuralnet)
    opt = RADAM()

    loss(x, y) = sum(Flux.Losses.binarycrossentropy(neuralnet(x), y))

    para = Flux.params(neuralnet)

    #=========Training the Model========#

    totalLoss = []
    traceY = []
    traceY2 = []

    for i in 1:epochs
        Flux.train!(loss, para, newTrainData, opt)
        println(i)
    end


    tempreds = []
    for (x,y) in newTestData
        push!(tempreds, neuralnet(x)[1])
    end

    push!(trueValues, classTest)
    push!(predictions, tempreds)
    push!(errors, 1-sum((tempreds .> .5) .== classTest)/length(classTest))
    println("round:", length(errors), ":", errors[end])

end

JLD.save("FFNNtrueValues.jld", "trueValues", trueValues)
JLD.save("FFNNpredictions.jld", "predictions", predictions)
JLD.save("FFNNerrors.jld", "errors", errors)

errors = JLD.load("FFNNerrors.jld", "errors")

best = argmin(errors)
worst = argmax(errors)

bestp = predictions[best]
worstp = predictions[worst]
bestt = trueValues[best]
worstt = trueValues[worst]
averageerr = mean(errors)

arbestp = convert(Vector{Float64}, bestp)
arworstp = convert(Vector{Float64}, worstp)
rocnumsbest = MLBase.roc(bestt.==1, arbestp, 50)
rocnumsworst = MLBase.roc(worstt.==1, arworstp, 50)

bestTPR = true_positive_rate.(rocnumsbest)
bestFPR = false_positive_rate.(rocnumsbest)
Plots.plot(bestFPR, bestTPR)

worstTPR = true_positive_rate.(rocnumsworst)
worstFPR = false_positive_rate.(rocnumsworst)
Plots.plot(bestFPR, bestTPR, label = "Best")
Plots.plot!(worstFPR, worstTPR, label = "Worst")
Plots.title!("ROC Curve, Dropout with Synthetic Data")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")
