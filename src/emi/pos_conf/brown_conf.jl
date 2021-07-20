using SparseArrays
using SparseArrayKit
using JLD
using Word2Vec
using LinearAlgebra
using Statistics
using Embeddings
using Flux
using Random
using DataFrames
using Lathe.preprocess: TrainTestSplit
using Plots
using StatsBase
using BSON
using ProgressBars
using CSV

# ------------------------ Data Cleaning ------------------------ #

include("brown_functions.jl")

# Reading in text file
brown_df = CSV.read("brown.csv", DataFrame)
brown_data = brown_df[4]
raw_sentences = split.(brown_data, " ")

raw_tags = []
raw_words = []
for sent in raw_sentences
    raw_tags_temp = []
    raw_words_temp = []
    for word in sent
        ind = findlast(x -> x == '/', word)
        push!(raw_tags_temp, word[ind+1:end])
        push!(raw_words_temp, lowercase(word[1:ind-1]))
    end
    push!(raw_tags, raw_tags_temp)
    push!(raw_words, raw_words_temp)
end




sentences = brown_df[5]
tags = brown_df[6]

# Splitting up sentences and tags
sentences = split.(sentences, " ")
tags = split.(tags, " ")

# Finding individual words
words = []
individual_tags = []
for i in 1:length(sentences)
    for j in 1:length(sentences[i])
        push!(words, lowercase(sentences[i][j]))
    end
    for k in 1:length(tags[i])
        push!(individual_tags, (tags[i][k]))
    end
    println(i)
end

tagger = unique!(individual_tags)
tagger = remove_wrong_tags(words, individual_tags)

function remove_wrong_tags(words, tags)

    for i in 1:length(words)



# Finding unique words and embeddings for each
unique_words = convert(Vector{String},unique(words))

# Making all words lowercase
for i in 1:length(unique_words)
    unique_words[i] = lowercase(unique_words[i])
end

# Finding embeddings for each unique word
embeddings_glove = load_embeddings(GloVe{:en},4, keep_words=Set(unique_words))
embtable = Dict(word=>embeddings_glove.embeddings[:,ii] for (ii,word) in enumerate(embeddings_glove.vocab))

# Finding the words that have GloVe embeddings
keys_embtable = []
for i in keys(embtable)
    push!(keys_embtable, i)
end

# Finding the words that don't have GloVe embeddings
no_embeddings = []
for i in 1:length(unique_words)
    embed_count = 0
    for j in 1:length(keys_embtable)
        if unique_words[i] == keys_embtable[j]
            embed_count += 1
        end
    end
    if embed_count == 0
        push!(no_embeddings, unique_words[i])
    end
    println(i)
end

act_word, act_pos, new_sentences = word_masker(sentences, tags)
