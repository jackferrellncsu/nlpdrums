using JLD
using Embeddings
using DataFrames
using Lathe.preprocess: TrainTestSplit
using Flux
using LinearAlgebra
using Plots
using Statistics
using Random
using StatsBase
using InvertedIndices
using BSON

#----------Prepare Embeddings---------------------------------------#
obj = load("PridePrej.jld")
    data = obj["data"]
    sentences = obj["sentances"]
    corpus = obj["corpus"]

embtable = load("pridePrejEmbs.jld", "embtable")
get_vector_word = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab))

vec_length = length(embtable.embeddings[:, get_word_index["the"]])

x_mat, y_mat = EmbeddingsTensor(data)

#split into test, proper_train, calibrate
train_x, test_x, train_y, test_y = SampleMats(x_mat, y_mat)
proper_train_x, calibrate_x, proper_train_y, calibrate_y = SampleMats(train_x, train_y)

trainDL = Flux.Data.DataLoader((proper_train_x, proper_train_y),
                            batchsize = 100,
                            shuffle = true)

calibrateDL = Flux.Data.DataLoader((calibrate_x, calibrate_y))

nn = Chain(Flux.flatten,
           Dense(1500, 800, mish),
           Dense(800, 500, mish),
           Dense(500, 300, x->x))

opt = RADAM(1e-4)
ps = Flux.params(nn)

epochs = 10
trace = TrainNN!(epochs)

plot(1:epochs, trace)

acc = 0
for (x, y) in calibrateDL
    acc += norm(y - nn(x))^2
end
mse_acc = acc / length(calibrateDL.data[2][1, :])

using BSON: @save
@save "basic.bson" nn

using BSON: @load

BSON.@load "basic.bson" nn



#-----------------NN Helpers---------------------------------------#
function loss(x, y)
    return norm(nn(x) - y)
end

function TrainNN!(epochs)
    traceY = []
    for i in 1:epochs
        Flux.train!(loss, ps, trainDL, opt)
        println(i)
        totalLoss = 0
        for (x,y) in trainDL
         totalLoss += loss2(x,y)
         #println("Total Loss: ", totalLoss)
        end
        push!(traceY, totalLoss)
    end
    return traceY
end



#---------Create Tensor--------------------------------------------#
function EmbeddingsTensor(data, context_size = 5)
    tensor = zeros(300, context_size, size(data)[1])

    result = zeros(300, size(data)[1])

    for (i, r) in enumerate(eachrow(data))
        sentence_mat = zeros(300, context_size)
        if length(r[1]) >= context_size
            for (j, w) in enumerate(r[1][end-4:end])
                sentence_mat[:, j] = getEmbedding(w)
            end
        else
            sent_length = length(r[1])
            for j in 1:context_size
                if j <= sent_length
                    sentence_mat[:, j] = getEmbedding(r[1][j])
                else
                    sentence_mat[:, j] = mean.(eachrow(sentence_mat[:, 1:sent_length]))
                end
            end
        end
        tensor[:, :, i] = sentence_mat
        result[:, i] = getEmbedding(r[2])
    end
    return tensor, result
end

#returns trainx, testx, trainy, testy
function SampleMats(x_mat, y_mat, prop = 0.9)
    inds = [1:size(x_mat)[3];]
    length(inds)
    trains = sample(inds, Int(floor(length(inds) * prop)), replace = false)
    inds = Set(inds)
    trains = Set(trains)
    tests = setdiff(inds, trains)

    train_x = x_mat[:, :, collect(trains)]
    train_y = y_mat[:, collect(trains)]

    test_x = x_mat[:, :, collect(tests)]
    test_y = y_mat[:, collect(tests)]


    return train_x, test_x, train_y, test_y
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
