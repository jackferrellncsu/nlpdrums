using Lathe
using Flux
using Word2Vec
include("../embeddings_nn.jl")
include("../data_cleaning.jl")

text = importClean()
#import text, sort by field, and create corpus
B = 50
errors = Vector{Float64}()
predictions = Vector{Vector{Float64}}()
trueValues = Vector{Vector{Float64}}()
for i in 1:B
    train, test = TrainTestSplit(text, 0.9)
    sort!(train, "medical_specialty")
    createCorpusText(train, 0)

    field = " Cardiovascular / Pulmonary"

    word2vec("corpus.txt", "vectors.txt", size = 15, window = 20)
    rm("corpus.txt")

    m = wordvectors("vectors.txt", normalize = true)

    train = filtration(train, field)

    classTrain = train[:,1].== field
    classTest = test[:,1].== field

    classTest *= 1.0
    classTrain *= 1.0

    vecsTrain = zeros(length(classTrain), 15)
    vecsTest = zeros(size(classTest)[1], 15)

    for i in 1:length(classTrain)
        vecsTrain[i, :] = formulateText(m, train[i,3])
    end

    for i in 1:size(classTest)[1]
        vecsTest[i, :] = formulateText(m, test[i, 3])
    end

    train_mat = vecsTrain'
    test_mat = vecsTest'

    trainingdata = Flux.Data.DataLoader((train_mat, classTrain'), batchsize = 100, shuffle = true)
    testingdata = Flux.Data.DataLoader((test_mat, classTest'))

    function neural_net()
        nn = Chain(
            Dense(15, 7, mish),
            Dropout(0.6),
            Dense(7, 1, x->σ.(x))
        )
    end

    neuralnet = neural_net()
    Flux.testmode!(neuralnet)
    opt = RADAM()

    lozz(x, y) = sum(Flux.Losses.binarycrossentropy(neuralnet(x), y))

    para = Flux.params(neuralnet)

    epochs = 1000
    for i in 1:epochs
        Flux.train!(lozz, para, trainingdata, opt)
    end


    # prints predictions vs actual
    # and error rate
    acc = 0
    tempreds = []
    for (x, y) in testingdata
        acc+=sum((neuralnet(x).>0.5) .== y)
        push!(tempreds, neuralnet(x)[1])
    end

    println(1 - acc/length(classTest))

    push!(errors, 1 - acc/length(classTest))
    push!(predictions, tempreds)
    push!(trueValues, classTest)

end

mean(errors)
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
Plots.title!("ROC Curve, Feed-Forward with Dropout")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")

#= Trace Plot ??
x = 1:epochs
y = traceY
Plots.plot(x, y, label = "Loss Progression")
xlabel!("Total # of epochs")
ylabel!("Loss values")
=#
