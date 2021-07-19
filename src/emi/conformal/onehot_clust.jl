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

include("pride_functions.jl")

# ------------------------ Data Cleaning ------------------------ #

# Reading in text file
###og_text = open("/Users/eplanch/Downloads/1342-0.txt", "r")
###corpus = read(og_text, String)
###close(og_text)

#pride_jld_creator(corpus)

# ------------------------- Loading File ------------------------ #

# Loads the PridePrej JLD file in
###pride_jld = JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/PridePrej.jld")
pride_jld = JLD.load("PridePrej.jld")
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
data = Matrix(data)

pre_sentence = data[:, 1]
next_word = data[:, 2]

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

# connect one hot representation to
y_class = BitArray(undef, 6135, size(data)[1])
for i in 1:size(data)[1]
    y_class[:, i]  = (Flux.onehot(data[i, 2], unique_words) .== 1)
end

# Splitting tensor into train/test/calib
# Takes a minute or two
train_tens_raw, test_tens_raw, calib_tens_raw = split_tensor(tensor, nextword_emb, .9, .9)

function get_onehot(tensor, y_class)

    one_hots = BitArray(undef, 6135, length(tensor[1, end, :]))
    for i in 1:length(tensor[1, end, :])
        word = get(embtable_back, tensor[:, end, i], 0)
        for j in 1:length(next_word)
            if word == next_word[j]
                one_hots[:,i] .= y_class[:,j]
            end
        end
        println(i)
    end

    return one_hots
end

# Matrices for classification
train_tens_class = get_onehot(train_tens_raw, y_class)
test_tens_class = get_onehot(test_tens_raw, y_class)
calib_tens_class = get_onehot(calib_tens_raw, y_class)

# Tensors for convolution
train_tens = train_tens_raw[1:300, 1:5, :]
test_tens = test_tens_raw[1:300, 1:5, :]
calib_tens = calib_tens_raw[1:300, 1:5, :]

# Convolution
conv_train = convolute_channel(train_tens, 3, relu) |> gpu
conv_test = convolute_channel(test_tens, 3, relu) |> gpu
conv_calib = convolute_channel(calib_tens, 3, relu) |> gpu

# Creation of DataLoader objects
dl_calib = Flux.Data.DataLoader((conv_calib, calib_tens_class))
dl_test = Flux.Data.DataLoader((conv_test, test_tens_class))
dl_train = Flux.Data.DataLoader((conv_train, train_tens_class),
                                    batchsize = 100, shuffle = true)

# ----------------------- Neural Net ----------------------- #

# Neural Net Architecture
nn = Chain(
    Dense(900, 3000, relu),
    Dense(3000, 6135, relu),
    softmax
    ) |> gpu

# Optimizer
opt = RADAM()

# Parameters
ps = Flux.params(nn)

# Loss Function
function loss(x, y)
    return Flux.Losses.crossentropy(nn(x), y)
end

# Training the Neural Net, Tracking Loss Progression
totalLoss = []
traceY = []
for i in 1:30
    Flux.train!(loss, ps, dl_train, opt)
    totalLoss = 0
end

JLD.save("dl_calib.jld", "dl_calib", dl_calib)
JLD.save("dl_test.jld", "dl_test", dl_test)
JLD.save("dl_train.jld", "dl_train", dl_train)

# Saving Model
nn |> cpu
using BSON: @save
@save "onehot_emi.bson" nn
