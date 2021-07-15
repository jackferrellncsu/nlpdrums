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
using StatsPlots

Random.seed!(26)
#----------Prepare Embeddings---------------------------------------#
obj = load("PridePrej.jld")
    data = obj["data"]
    sentences = obj["sentances"]
    corpus = obj["corpus"]

embtable = load("pridePrejEmbs.jld", "embtable")
#get vector from word
get_vector_word = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab))
#get word from vector
get_word_vector = Dict(embtable.embeddings[:, ii] => word for (ii, word) in enumerate(embtable.vocab))
#get index from word
get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
vec_length = length(embtable.embeddings[:, get_word_index["the"]])

#filter out non-embedded outcomes
unique_words = [word for word in keys(get_vector_word)]
data = DataFrame(data)
filter!(row -> row[2] ∈ unique_words, data)
data = Matrix(data)

x_mat, y_mat = EmbeddingsTensor(data, 5)

#split into test, proper_train, calibrate
train_x, test_x, train_y, test_y = SampleMats(x_mat, y_mat)
proper_train_x, calibrate_x, proper_train_y, calibrate_y = SampleMats(train_x, train_y, .92)

trainDL = Flux.Data.DataLoader((proper_train_x, proper_train_y),
                            batchsize = 1000,
                            shuffle = true)

calibrateDL = Flux.Data.DataLoader((calibrate_x, calibrate_y))
testDL = Flux.Data.DataLoader((test_x, test_y))

nn = Chain(Flux.flatten,
           Dense(1500, 800, mish),
           Dense(800, 500, mish),
           Dense(500, 300, x->x))

opt = RADAM(1e-4)


epochs = 20
trace,mse  = TrainNN!(epochs, loss, nn, opt)


plot(1:epochs, mse; label = "Validation MSE")

err = 0
for (x, y) in calibrateDL
    err += norm(y - nn(x))^2
end
mse_acc = err / length(calibrateDL.data[2][1, :])

#Save and load model
using BSON: @save
@save "basic.bson" nn

using BSON: @load

BSON.@load "basic.bson" nn


test = ConfPred(nn)


dist_from_the = Dict()
for (x, y) in testDL
    dist = norm(nn(x) - get_vector_word["the"])
    dist_from_the[get_word_vector[y]] = dist
end

accuracy = CheckValidity(test)
mean_size, med_size = IntervalEfficiency(test)



"""
    CheckValidity(intervals)

Checks how many values actually lie in the confidence regions generated by ICP
"""
function CheckValidity(intervals)
    acc = 0
    for (i, region) in enumerate(intervals)
        if mean(test_y[:,i]) != 0 && get_word_vector[test_y[:, i]] ∈ region
            acc += 1
        end
    end
    return acc / length(intervals)

end

function IntervalEfficiency(intervals)
    lengths = length.(intervals)
    return median(lengths), mean(lengths)
end

"""
    ConfPred(nn, ϵ = 0.05)

Given a neural net, performes inductive conformal prediction and returns prediction
regions for test set.
"""
function ConfPred(nn, ϵ = 0.05)
    α = Vector{Float64}()
    for (x, y) in calibrateDL
        α_i = norm(y - nn(x))
        push!(α, α_i)
    end
    println("Noncomformity scores calculated")
    all_regs = Vector{Vector{String}}()
    for (x, y) in testDL
        a_k = norm(y - nn(x))
        push!(α, a_k)
        q = quantile(α, 1-ϵ)
        region = Vector{String}()
        pred = nn(x)
        for i in get_vector_word
            dist = norm(pred - i[2])
            if dist <= q
                push!(region, i[1])
            end
        end
        pop!(α)
        push!(all_regs, region)
    end
    return all_regs
end


#-----------------NN Helpers---------------------------------------#
"""
    loss(x, y)

Naive loss for regressive next word prediction, calculated as Euclidean distance
between observed and predicted.
"""
function loss(x, y)
    return norm(nn(x) - y)
end

function loss2(x, y)
    return norm(nn(x)-y)
end

"""
    TrainNN!(epochs, loss, nn, opt)

Trains a neural net using the specified number of epochs.  Uses input optimizer
and loss function.

Returns a trace plot.
"""
function TrainNN!(epochs, loss, nn, opt)
    traceY = []
    mse = []

    ps = Flux.params(nn)

    for i in 1:epochs
        Flux.train!(loss, ps, trainDL, opt)
        println(i)
        totalLoss = 0
        for (x,y) in trainDL
         totalLoss += loss(x,y)
         #println("Total Loss: ", totalLoss)
        end
        err = 0
        for (x, y) in calibrateDL
            err += norm(y - nn(x))^2
        end
        mse_acc = err / length(calibrateDL.data[2][1, :])
        push!(traceY, totalLoss)
        push!(mse, mse_acc)
    end
    return traceY, mse
end

#---------Create Tensor--------------------------------------------------------#
"""
    EmbeddingsTensor(data, context_size = 5)

Creates a 3-dimensional matrix of word embeddings
"""
function EmbeddingsTensor(data, context_size = 5)
    tensor = zeros(300, context_size, size(data)[1])

    result = zeros(300, size(data)[1])

    for (i, r) in enumerate(eachrow(data))
        sentence_mat = zeros(300, context_size)
        if length(r[1]) >= context_size && context_size != 1
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

#-----------------Diagnostics--------------------------------------------------#
"""
    DistToAll(word)

Calculates distance between given word and all others from corpus in GloVe vector
space.
"""
function DistToAll(word)
    word_vec = get_vector_word[word]
    dist_dict = Dict()
    for iter in get_vector_word
        dist = norm(word_vec - iter[2])
        dist_dict[iter[1]] = dist
    end
    return dist_dict

end


means = []
for word in keys(get_vector_word)
    temp = DistToAll(word)
    push!(means, mean(values(temp)))
end

mean(means)

density(means, leg = false, color = :black, fill = (0, 0.5, RGB(254 / 264.0, 0, 0)))
title!("Distribution of Average Distance Between All Other Words")
vline!([mean(means)], line = (4, :dash, 0.6, [:blue]), label = "Mean")
xlabel!("Average Distance Between Given Word and All Others")
ylabel!("Proportion of Occurences")
png("AvgGloVeDist_Final")
