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

# --------------------- Data Cleaning/Loading --------------------- #

include("pride_functions.jl")

# Reading in text file
og_text = open("/Users/eplanch/Downloads/1342-0.txt", "r")
corpus = read(og_text, String)
close(og_text)

pride_jld_creator(corpus)

# Loads the PridePrej JLD file in
pride_jld = JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/conformal/PridePrej.jld")
embedding_table = pride_jld["embtable"]
word_index = pride_jld["word_index"]
embtable_back = pride_jld["embtable_back"]
corp = pride_jld["corpus"]
split_sent = pride_jld["split_sentences"]
data = pride_jld["data"]

# Removing words with no embedding
unique_words = [word for word in keys(embedding_table)]
data = DataFrame(data)
filter!(row -> row[2] ∈ unique_words, data)
filter!(row -> length(row[1]) > 2, data)
data = Matrix(data)

pre_sentence = data[:, 1]
next_word = data[:, 2]

# ----------------------- Data Prep for NN ----------------------- #

# Reversing order of each pre-sentence
for i in 1:length(pre_sentence)
    pre_sentence[i] = reverse(pre_sentence[i])
end

# Creating word embeddings for each "next word" after the pre-sentences
nextword_emb = zeros(300, length(next_word))
for i in 1:length(next_word)
    nextword_emb[:, i] = get(embedding_table, next_word[i], zeros(300))
    println(i)
end

# Creating the tensor for the pre-sentence embeddings
tensor = create_tensor(embedding_table, pre_sentence, 300, 5)

# Splitting tensor into train/test/calib
train_tens_raw, test_tens_raw, calib_tens_raw = split_tensor(tensor, nextword_emb, .9, .9)

train_tens, train_tens_class = data_class_split(train_tens_raw)
test_tens, test_tens_class = data_class_split(test_tens_raw)
calib_tens, calib_tens_class = data_class_split(calib_tens_raw)

# Convolution
conv_train = mat2vec(convolute_channel(train_tens, 2, relu))
conv_test = mat2vec(convolute_channel(test_tens, 2, relu))
conv_calib = mat2vec(convolute_channel(calib_tens, 2, relu))



#=
# Creation of DataLoader objects
dl_calib = Flux.Data.DataLoader((conv_calib, calib_tens_class))
dl_test = Flux.Data.DataLoader((conv_test, test_tens_class))
dl_train = Flux.Data.DataLoader((conv_train, train_tens_class),
                                    batchsize = 100, shuffle = true)
=#

# ----------------------- Neural Net ----------------------- #

# Layers of rnn
L1 = length(conv_train[1])
LE = 6135
# Neural Net Architecture
rnn = Chain(
    Flux.GRU(L1, 1000),
    Dense(1000, LE, relu),
    softmax)

# Optimizer
opt = RADAM()

# Parameters
ps = Flux.params(rnn)

# Loss Function
function loss(x, y)
    Flux.reset!(rnn)
    return Flux.Losses.crossentropy(rnn.(x), y)
end

evalcb() = @show(sum(loss.(conv_test, test_tens_class)))

# Training the Neural Net, Tracking Loss Progression
totalLoss = []
traceY = []
for i in ProgressBar(1:5)
    Flux.train!(loss, ps, zip(conv_train, train_tens_class), opt)
    totalLoss = 0
    for (x,y) in dl_train
        totalLoss += loss(x,y)
    end
    push!(traceY, totalLoss)
end

# Saving Model
using BSON: @save
@save "rnn.bson" rnn
