using Lathe
using Flux
using Word2Vec
using MLBase
using Plots
include("../embeddings_nn.jl")
include("../data_cleaning.jl")

Random.seed!(13)

text = importClean()
#import text, sort by field, and create corpus
B = 50
errors = Vector{Float64}()
predictions = Vector{Vector{Float64}}()
trueValues = Vector{Vector{Float64}}()
for i in 1:1
    train, test = TrainTestSplit(text, 0.7)
    sort!(train, "medical_specialty")
    createCorpusText(train, 0)

    field = " Cardiovascular / Pulmonary"

    word2vec("corpus.txt", "vectors.txt", size = 300, window = 20)
    rm("corpus.txt")

    m = wordvectors("vectors.txt", normalize = true)

    train = filtration(train, field)

    classTrain = train[:,1].== field
    classTest = test[:,1].== field

    classTest *= 1.0
    classTrain *= 1.0

    vecsTrain = zeros(length(classTrain), 300)
    vecsTest = zeros(size(classTest)[1], 300)

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

    d = 0.7

    function neural_net()
        nn = Chain(
            Dense(300, 150, mish),
            Dropout(d),
            Dense(150, 50, mish),
            Dropout(d),
            Dense(50, 10, mish),
            Dropout(d),
            Dense(10, 1, x->σ.(x))
            )
        end

    neuralnet = neural_net()
    Flux.testmode!(neuralnet)
    opt = RADAM()

    lozz(x, y) = sum(Flux.Losses.binarycrossentropy(neuralnet(x), y))

    para = Flux.params(neuralnet)

    epochs = 500
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


plotly()
#Plots.plot(1:5,1:5)
mean(errors)


roc_nums = roc((classTest.==1), convert(Vector{Float64}, tempreds))

tpr = true_positive_rate.(roc_nums)
fpr = false_positive_rate.(roc_nums)

Plots.plot(fpr, tpr)
Plots.title!("Feed-Forward Net with Dropout, Real Data")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")
