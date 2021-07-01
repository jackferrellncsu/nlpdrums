using GLM
using JLD
using CSV
using DataFrames
using Embeddings
using LinearAlgebra
using MLBase
using Plots

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

embtable = load("embtable_word2vec.jld", "embtable")
get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
vec_length = length(embtable.embeddings[:, get_word_index["the"]])


trainEmb = SampleEmbeddings(train, vec_length)
testEmb = SampleEmbeddings(test, vec_length)

classTrain = train[:, 2]
classTest = test[:, 2]

X = convert(Matrix{Float64}, trainEmb')

model = glm(X, classTrain, Binomial(), LogitLink())

newX = convert(Matrix{Float64}, testEmb')

preds = predict(model, newX)

err_rat = 1 - sum(preds .== classTest) / 3000

roc_nums = roc((classTest .== 1), preds)

tpr = true_positive_rate.(roc_nums)
fpr = false_positive_rate.(roc_nums)

plot(fpr, tpr, label = "Error", yaxis = [0, 1])
title!("Word2Vec Logit ROC")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")

png("word2vec_synth_logit_roc")
