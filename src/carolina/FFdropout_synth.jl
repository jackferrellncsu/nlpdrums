using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--opt1"
            help = "an option with an argument"
        "--opt2", "-o"
            help = "another option with an argument"
            arg_type = Int
            default = 0
        "--flag1"
            help = "an option without argument, i.e. a flag"
            action = :store_true
        "arg1"
            help = "a positional argument"
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

using Flux
using LinearAlgebra
using Statistics
using CSV
using Random
using Embeddings
using Lathe.preprocess: TrainTestSplit
using JLD
using DataFrames

# veclength = # of input nodes

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

vec_length = 300

TrainTestSplitPercent = 0.7
epochs = 500

#n = parse(Int64, get(parsed_args, "arg1", 0))
n = 1
Random.seed!(n%100)
param = Int(ceil(n/100))

dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
d = dropouts[param]

obj = load("embtable_word2vec.jld")
embtable = obj["embtable"]

get_word_index = Dict(word => ii for (ii, word) in enumerate(embtable.vocab))

data = CSV.read("../JonsTraining.csv", DataFrame)

#train, test = TrainTestSplit(data[(1-1)*1000+1:(1)*1000, :])
#SampleEmbeddings(train, 300)


train, test = TrainTestSplit(data, TrainTestSplitPercent)
trainEmbs = SampleEmbeddings(train, vec_length)
testEmbs = SampleEmbeddings(test, vec_length)



classTrain = train[:,2]
classTrain = classTrain * 1.0

classTest = test[:,2]
classTest = classTest * 1.0

tmat = Matrix(test)

#=
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
=#
# =============Neural Net Stuff=============#

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

totalLoss = []
traceY = []
traceY2 = []

for k in 1:epochs
    Flux.train!(loss, para, newTrainData, opt)
    if k%100 == 0
        println(k)
    end
end


tempreds = []
for (x,y) in newTestData
    push!(tempreds, neuralnet(x)[1])
end

#push!(trueValues, classTest)
#push!(predictions, tempreds)
errors = 1-sum((tempreds .> .5) .== classTest)/length(classTest)


#JLD.save("FFNNtrueValues.jld", "trueValues", trueValues)
#JLD.save("FFNNpredictions.jld", "predictions", predictions)
JLD.save("FFNNerrors" * string(param)* "_"* string(n%100)*".jld", "errors", errors)
