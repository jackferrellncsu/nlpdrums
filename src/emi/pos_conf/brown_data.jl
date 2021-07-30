using JLD
using LinearAlgebra
using Statistics
using Embeddings
using Flux
using Flux: Losses
using Random
using DataFrames
using Plots
using StatsBase
using BSON
using ProgressBars
using CSV

Random.seed!(24)

println("Phase 1 Complete")

include("brown_functions.jl")

# Reading in text file
brown_df = CSV.read("brown.csv", DataFrame)
brown_data = brown_df[4]
raw_sentences = split.(brown_data, " ")

# Finding unique words and embeddings for each
unique_pos, sentences, sentence_tags = data_cleaner(raw_sentences)
words = get_word_vec(sentences)
unique_words = convert(Vector{String},unique(words))

# Finding embeddings for each unique word
embeddings_glove = load_embeddings(GloVe{:en},4, keep_words=Set(unique_words))
embtable = Dict(word=>embeddings_glove.embeddings[:,ii] for (ii,word) in enumerate(embeddings_glove.vocab))

# Finding the words that have GloVe embeddings
keys_embtable = [keyz for keyz in keys(embtable)]

# Finding the words that don't have GloVe embeddings
no_embeddings = setdiff(unique_words, keys_embtable)

# Creating a tensor of word embeddings
sent_tens, new_sent, new_tags = sent_embeddings(sentences, sentence_tags, 300, 180, embtable)

# Converting outputs of the above function
new_sent = convert(Vector{Vector{String}}, new_sent)
new_tags = convert(Vector{Vector{String}}, new_tags)
sent_tens = convert(Array{Float32, 3}, sent_tens)

# Masks random word in each sentence
masked_word, masked_pos, new_sentences = word_masker(new_sent, new_tags)

# Finds the indices of each masked word and fills tensor with -20.0's at those
# indices
mask_ind, sent_tens_emb = masked_embeddings(new_sentences, sent_tens, 300)

# Centers masked word in tensor and truncates length of sentence
sent_tens_emb = create_window(sent_tens_emb, 15)

# Converting output again
sent_tens_emb = convert(Array{Float32, 3}, sent_tens_emb)

# Creating one hot matrix for the tensor
onehot_vecs = zeros(length(unique_pos), length(masked_pos))
for i in 1:length(masked_pos)
    onehot_vecs[:, i] = Flux.onehot(masked_pos[i], unique_pos)
end

# Converting the one hot matrix
onehot_vecs = convert(Array{Float32, 2}, onehot_vecs)

# Save JLD
JLD.save("brown_data.jld", "onehots", onehot_vecs,
                           "sentence_tensor", sent_tens_emb,
                           "unique_pos", unique_pos)
