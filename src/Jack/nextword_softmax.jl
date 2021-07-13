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

Random.seed!(26)

#Embeddings prep etc
obj = load("PridePrej.jld")
    data = obj["data"]
    sentences = obj["sentances"]
    corpus = obj["corpus"]

embtable = load("pridePrejEmbs.jld", "embtable")
#get vector from word
get_vector_word = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab))
#get word from vector
get_word_vector = Dict(embtable.embeddings[:, ii] => word for (ii, word) in enumerate(embtable.vocab))
get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
vec_length = length(embtable.embeddings[:, get_word_index["the"]])

unique_words = [word for word in keys(get_vector_word)]
data = DataFrame(data)
filter!(row -> row[2] ∈ unique_words, data)
data = Matrix(data)


x_mat, trash = EmbeddingsTensor(data)
y_mat = Flux.onehotbatch(data[:, 2], unique_words)



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

function OneHots(data, labels)
    oneHotMat = Flux.onehotbatch(data[:, 2], labels)
end

OneHots(data, unique_words)
