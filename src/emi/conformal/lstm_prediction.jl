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
#filter!(row -> length(row[1]) > 2, data)
data = Matrix(data)

# Load in pre sentences and next words
pre_sentence = data[:, 1]
next_word = data[:, 2]

# ----------------------- Data Prep for NN ----------------------- #


# Creating word embeddings for each "next word" after the pre-sentences
nextword_emb = zeros(300, length(next_word))
for i in 1:length(next_word)
    nextword_emb[:, i] = get(embedding_table, next_word[i], zeros(300))
    println(i)
end
nextword_emb = Matrix{Float32}(nextword_emb[:, 2:end])


# Creating a corrected length of pre-sentences
pre_sentence = Matrix{Float32}(nextword_emb[:, 1:end-1])

# Splitting data into data/class; class is in one-hot representation
function split_classes(matrix, next_word, length, train_test, train_calib, unique_words)

    # Computing sizes of each set
    b = matrix[1,:]
    a = length(b)
    L = length(matrix[1,:]) * train_test
    first_train_size = Int(ceil(L))
    test_size = Int(length(matrix[1,:]) - first_train_size)
    train_size = Int(ceil(first_train_size * train_calib))
    calib_size = Int(first_train_size - train_size)

    train = matrix[:, 1:train_size]
    train_class = create_class(train, next_word, length, unique_words)

    test = matrix[:, train_size+1:train_size+test_size]
    test_class = create_class(test, next_word, length, unique_words)

    calib = matrix[:, test_size+train_size+1:train_size+test_size+calib_size]
    calib_class = create_class(calib, next_word, length, unique_words)

    return train, train_class, test, test_class, calib, calib_class
    end
    train, train_class, test, test_class, calib, calib_class = split_classes(pre_sentence, next_word, 6135, .9, .9, unique_words)

# Creation of DataLoader objects
dl_calib = Flux.Data.DataLoader((calib, calib_class))
dl_test = Flux.Data.DataLoader((test, test_class))
dl_train = Flux.Data.DataLoader((train, train_class),
                                    batchsize = 1709, shuffle = false)


# ----------------------- Neural Net ----------------------- #

# Layers of rnn
L1 = 300
LE = 6135

# Neural Net Architecture
rnn = Chain(
    Flux.LSTM(L1, 2000),
    Flux.LSTM(2000, 2000),
    Dense(2000, LE, relu),
    softmax)

# Optimizer
opt = RADAM()

# Parameters
ps = Flux.params(rnn)

# Loss Function
function loss(x, y)
    return Flux.Losses.crossentropy(rnn(x), y)
end

# Training the Neural Net, Tracking Loss Progression
epochs = 5
traceY = []
for i in 1:epochs
    println("Starting epoch #", i, " ...")
    Flux.reset!(rnn)
    Flux.train!(loss, ps, dl_train, opt)
    Flux.reset!(rnn)
    L = sum(loss.(eachcol(pre_sentence[:,1:5]), eachcol(y_class[:,1:5])))
    push!(traceY, L)
end

JLD.save("dl_calib.jld", "dl_calib", dl_calib)
JLD.save("dl_test.jld", "dl_test", dl_test)
JLD.save("dl_train.jld", "dl_train", dl_train)

# Saving Model
using BSON: @save
@save "rnn_five_epoch.bson" rnn

# Conformal predictions
setto = Vector{String}()
setto = inductive_conformal(rnn, 0.05, dl_test)
