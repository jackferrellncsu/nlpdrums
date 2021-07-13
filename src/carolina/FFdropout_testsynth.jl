using Flux
using LinearAlgebra
using Statistics
using CSV
using Random
using Embeddings
using Lathe.preprocess: TrainTestSplit
using JLD
using DataFrames
using MLBase


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

# veclength = # of input nodes
vec_length = 300

TrainTestSplitPercent = 0.7
epochs = 500

#n = parse(Int64, get(parsed_args, "arg1", 0))
#Random.seed!(n%100)
#param = Int(ceil(n/100))

d = 0.7

obj = load("embtable_word2vec.jld")
embtable = obj["embtable"]

get_word_index = Dict(word => ii for (ii, word) in enumerate(embtable.vocab))

train = CSV.read("../JonsTraining.csv", DataFrame)
test = CSV.read("../JonsTest.csv", DataFrame)


trainEmbs = SampleEmbeddings(train, vec_length)
testEmbs = SampleEmbeddings(test, vec_length)

classTrain = train[:,2]
classTrain = classTrain * 1.0

classTest = test[:,2]
classTest = classTest * 1.0

tmat = Matrix(test)

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

newTrainData = Flux.Data.DataLoader((trainEmbs, classTrain'), batchsize = 100,
                                    shuffle = true)
newTestData = Flux.Data.DataLoader((testEmbs, classTest'))


neuralnet = neural_net()

opt = RADAM()

loss(x, y) = sum(Flux.Losses.binarycrossentropy(neuralnet(x), y))

para = Flux.params(neuralnet)

#=========Training the Model========#


for k in 1:epochs
    Flux.train!(loss, para, newTrainData, opt)
    if k%100 == 0
        println(k)
    end
end

#testmode!(neuralnet)

tempreds = []
for (x,y) in newTestData
    push!(tempreds, neuralnet(x)[1])
end

#push!(trueValues, classTest)
#push!(predictions, tempreds)
errors = 1-sum((tempreds .> .5) .== classTest)/length(classTest)


roc_nums = roc((classTest.==1), convert(Vector{Float64}, tempreds))

tpr = true_positive_rate.(roc_nums)
fpr = false_positive_rate.(roc_nums)

Plots.plot(fpr, tpr)
Plots.title!("Feed-Forward Net with Dropout, Synthetic Data")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")
