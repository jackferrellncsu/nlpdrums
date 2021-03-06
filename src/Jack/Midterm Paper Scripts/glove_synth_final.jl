using Glowe
using Lathe.preprocess: TrainTestSplit
using Embeddings
using LinearAlgebra
using Flux
using Plots
using JLD
using Random
using CSV
using DataFrames



function SampleEmbeddings(df, vec_size)
    embed = 0
    embed_mat = Matrix{Float64}(I, vec_size, length(eachrow(df)))
    for (i, r) in enumerate(eachrow(df))
        doc = split(r[1])
        embed = getEmbedding(doc[1])
        for d in doc[2:end]
            embed += getEmbedding(d)
        end
        embed_mat[:, i] = embed
    end

    return embed_mat
end

function getEmbedding(word)
    if word ∈ keys(get_word_index)
        ind = get_word_index[word]
        emb = embtable.embeddings[:, ind]
    else
        vec_length = length(embtable.embeddings[:, get_word_index["the"]])
        emb = zeros(vec_length)
    end
    return emb
end


train = CSV.read("../JonsTraining.csv", DataFrame)
test = CSV.read("../JonsTest.csv", DataFrame)

errorrates = []
predictions = []
trueValues = []

obj = load("embtable.jld")
#Chose 300D/42B based on CV
embtable = obj["embtable3"]

get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
vec_length = length(embtable.embeddings[:, get_word_index["the"]])

trainEmbs = SampleEmbeddings(train, vec_length)
testEmbs = SampleEmbeddings(test, vec_length)

#Get classifications for train/val/test
classTrain = train[:, 2]
classTest = test[:, 2]

batchsize_custom = 100

trainDL = Flux.Data.DataLoader((trainEmbs, classTrain'),
                                batchsize = batchsize_custom,
                                shuffle = true)
testDL = Flux.Data.DataLoader((testEmbs, classTest'))

reduc1 = Int(floor(vec_length*0.66))
reduc2 = Int(floor(reduc1*0.66))
reduc3 = Int(floor(reduc2*0.5))
nn = Chain(Dense(vec_length, reduc1, mish),
            Dense(reduc1, reduc2, mish),
            Dense(reduc2, reduc3, mish),
            Dense(reduc3, 10, mish),
            Dense(10, 1, x->Flux.σ.(x)))

opt = Flux.Optimiser(ExpDecay(0.01, 0.9, 200, 1e-4), RADAM())
ps = Flux.params(nn)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))

traceY = []
traceY2 = []
epochs = 200
    for i in 1:epochs
        Flux.train!(loss, ps, trainDL, opt)
        if i % 100 == 0
            println(i)
        end
        # for (x,y) in trainDL
        #     totalLoss = loss(x,y)
        # end
        # push!(traceY, totalLoss)
        # acc = 0
        # for (x,y) in testDL
        #     acc += sum((nn(x) .> .5) .== y)
        # end
        # decimalError = 1 - acc/length(classVal)
        # percentError = decimalError * 100
        # percentError = round(percentError, digits=2)
        # push!(traceY2, percentError)
    end

acc = 0
    for (x,y) in testDL
        global acc += sum((nn(x) .> .5) .== y)
    end

preds = (nn(testEmbs))
trues = classTest
errors = 1 - acc/length(classTest)

roc_nums = roc((trues .==1), vec(preds))

tpr = true_positive_rate.(roc_nums)
fpr = false_positive_rate.(roc_nums)

plot(fpr, tpr, leg = false)
title!("GloVe Embeddings Synthetic Data ROC")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")

png("GloVe_Synth_Roc")
