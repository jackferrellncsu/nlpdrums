using GLM
using JLD
using CSV
using DataFrames
using Embeddings
using LinearAlgebra
using Lathe.preprocess: TrainTestSplit
using MLBase
using Plots
using Random


include("../data_cleaning.jl")
include("../embeddings_nn.jl")


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

data = importClean()

train_pct = .7

embtable = load("embtable_word2vec.jld", "embtable")
get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
vec_length = length(embtable.embeddings[:, get_word_index["the"]])

data = filtration(data, " Cardiovascular / Pulmonary")

train, test = TrainTestSplit(data, train_pct)

classTrain = (train[:, 1] .== " Cardiovascular / Pulmonary") * 1.0
classTest = (test[:, 1] .== " Cardiovascular / Pulmonary") * 1.0

#Get embeddings for train/test
embtable = load("embtable_word2vec.jld", "embtable")

trainEmb = SampleEmbeddings(train, vec_length)
testEmb = SampleEmbeddings(test, vec_length)

X = convert(Matrix{Float64}, trainEmb')
X2 = convert(Matrix{Float64}, testEmb')

model = glm(X, classTrain, Binomial(), LogitLink())

newX = convert(Matrix{Float64}, testEmb')

preds = predict(model, newX)

err_rat = 1 - sum(preds .== classTest) / length(classTest)

roc_nums = roc((classTest .== 1), preds)

tpr = true_positive_rate.(roc_nums)
fpr = false_positive_rate.(roc_nums)

plot(fpr, tpr, label = "Error", leg = false)
xticks!(collect(0:0.2:1))
yticks!(collect(0:0.2:1))
title!("Word2Vec Logit ROC")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")

png("word2vec_real_logit_roc")
