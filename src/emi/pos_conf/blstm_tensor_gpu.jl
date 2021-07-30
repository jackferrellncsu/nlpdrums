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
brown_data = JLD.load("brown_data.jld")
sent_tens_emb = brown_data["sentence_tensor"]
onehot_vecs = brown_data["onehots"]

temp_train, test, temp_train_class, test_class = SampleMats(sent_tens_emb, onehot_vecs) |> gpu
train, calib, train_class, calib_class = SampleMats(temp_train, temp_train_class) |> gpu

println("Phase 2 complete")

# Creating DataLoader
dl_calib = Flux.Data.DataLoader((calib, calib_class))
dl_test = Flux.Data.DataLoader((test, test_class))
dl_train = Flux.Data.DataLoader((train[:, :, 1:10], train_class[:, 1:10]),
                                    batchsize = 100, shuffle = true)

# Neural net architecture
forward = LSTM(300, 150) |> gpu
backward = LSTM(300, 150) |> gpu
embedding = Dense(300, 300)|> gpu
predictor = Chain(Dropout(0.2, dims=1), Dense(300, 250, relu), Dense(250,190), softmax)|> gpu

# Predicts embedding for masked word
vectorizer(x) = embedding(BLSTM(x))

# Predicts pos tag based on vectorier output
model(x) = predictor(vectorizer(x))

# Optimizer
opt = Flux.Optimiser(ExpDecay(0.01, 0.1, 1, 1e-4), RADAM())

# Model parameters
ps = Flux.params((forward, backward, embedding, predictor))

# Cluster indicator
println("Beginning training")

# Training the neural net, tracking loss progression
epochs = 2
traceY = []
for i in ProgressBar(1:epochs)
    Flux.reset!(forward)
    Flux.reset!(backward)
    Flux.train!(loss, ps, dl_train, opt)
    @show i
    for (x, y) in dl_train
        push!(traceY, loss(x, y))
        break
    end
end

# Cluster indicator
println("Training complete")

# Saving everything needed from cluscpu
forward = forward |> cpu
backward = backward |> cpu
embedding = embedding |> cpu
predictor = predictor |> cpu

weights = []
push!(weights, Flux.params(forward))
push!(weights, Flux.params(backward))
push!(weights, Flux.params(embedding))
push!(weights, Flux.params(predictor))

using BSON: @save
BSON.@save "lstm_params_30.bson" weights
JLD.save("trace30.jld", "trace", traceY)
