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
get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
vec_length = length(embtable.embeddings[:, get_word_index["the"]])

#filter out non-embedded outcomes
unique_words = [word for word in keys(get_vector_word)]
data = DataFrame(data)
filter!(row -> row[2] ∈ unique_words, data)
data = Matrix(data)









#-----------------------Embeddings Helpers-------------------------------------#
function SummedContextEmbeddings(mat, vec_length)
    summed_embs = []
    res_embs = []
    for (i, r) in enumerate(eachrow(mat))
        con_emb = zeros(vec_length)
        res_emb = getEmbedding(r[2])
        for (j, word) in enumerate(r[1])
            con_emb += (1/2^(j-1)) .* getEmbedding(word)
        end
        push!(summed_embs, con_emb)
        push!(res_embs, res_emb)
    end
    println(length(summed_embs))
    println(length(res_embs))
    z = hcat(summed_embs, res_embs)
    return DataFrames.DataFrame(z)
end
