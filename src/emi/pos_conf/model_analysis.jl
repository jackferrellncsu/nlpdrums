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
using Zygote

Random.seed!(24)
include("brown_functions.jl")
# ----------------------- Loading data in for testing ----------------------- #

brown_data = JLD.load("brown_data.jld")
sent_tens_emb = brown_data["sentence_tensor"]
onehot_vecs = brown_data["onehots"]
unique_pos = brown_data["unique_pos"]

temp_train, test, temp_train_class, test_class = SampleMats(sent_tens_emb, onehot_vecs) |> gpu
train, calib, train_class, calib_class = SampleMats(temp_train, temp_train_class) |> gpu

# Creating DataLoader
dl_calib = Flux.Data.DataLoader((calib, calib_class))
dl_test = Flux.Data.DataLoader((test, test_class))
dl_train = Flux.Data.DataLoader((train, train_class),
                                    batchsize = 100, shuffle = true)

# -------------------------------- The Model -------------------------------- #

using BSON: @load
BSON.@load "lstm_params.bson" weights

forward = LSTM(300, 150)
backward = LSTM(300, 150)
embedding = Dense(300, 300)
predictor = Chain(Dense(300, 250, relu), Dense(250,190), softmax)

load_parameters!(weights)

vectorizer(x) = embedding(BLSTM(x))

trained_model(x) = predictor(vectorizer(x))

# -------------------------- Conformal Predictions -------------------------- #

function find_nonconf(dl_calib, model)
    α_i = Vector{Float64}()
    α = 0.0
    cor = 0.0
    for (x, y) in dl_calib
        cor = maximum(y .* model(x))
        α = 1 - cor
        push!(α_i, α)
        println(length(α_i)/length(dl_calib))
    end
    sort!(α_i, rev = true)
    return α_i
end

function find_ACDS(pvals, right_vec)
    oners = sum.(greatorVec.(pvals)) .== 1
    counter = 0
    for i in 1:length(oners)
        if oners[i] == 1
            if right_vec[i] == true
                counter += 1
            end
        end
    end
    return counter / sum(oners)
end

function find_PIS(pvals)
    pis = sum(sum.(greatorVec.(pvals)) .>= 2)/length(pvals)
    return pis
end

function model_accuracy(amount_right, pvals)
    return amount_right / length(pvals)
end

function test_blstm(dl_test, model)
    correct = []
    sets = []
    amount_right = 0
    for (x, y) in dl_test
        V = model(x)
        push!(sets, toPval(V, α_i))
        Y = []
        for i in y
            push!(Y, i)
        end
        push!(correct, Y)
        if argmax(V) == argmax(y)
            amount_right += 1
        end
    end
    return correct, sets, amount_right
end

function find_OP_OF(pvals, actWords)
    OP = mean(dot.(pvals,actWords))
    OF = mean(dot.(pvals, notVec.(actWords)) / length(pvals[1]))
    return OP, OF
end

# Creats the conformal sets
percent_right = 0.0
set_sizes = Vector{Int64}
sets = Vector{Vector{String}}
percent_right, right_vec, set_sizes, sets = inductive_conformal(trained_model, .01,
                                            dl_calib, dl_test, unique_pos)

# Finds average set size
avg_set = mean(set_sizes)

# Finds the empirical vs proposed validity
set_averages = []
confs = []
valid = []
for i in 0:.05:.5
    a, b, c = inductive_conformal(trained_model, i, dl_calib, dl_test, unique_pos)
    aver = mean(b)
    conf = 1 - i
    push!(set_averages, aver)
    push!(confs, conf)
    push!(valid, a)
end

# Finds all data for paper
α_i = find_nonconf(dl_calib, trained_model)
actWords, pvals, amount_right = test_blstm(dl_test, trained_model)
model_acc = model_accuracy(amount_right, pvals)
OP, OF = find_OP_OF(pvals, actWords)
pis = find_PIS(pvals)
acds = find_ACDS(pvals, right_vec)
conf = mean(maximum.(pvals))

# not sure if this is right
correct = sum(argmax.(actWords) .== argmax.(pvals)) /length(pvals)
credibility = mean(maximum.(pvals))

global epsilon = .01
sizes = sum.(greatorVec.(pvals))
ncrit = mean(sizes)
empconf = mean(returnIndex.(pvals, argmax.(actWords)) .> epsilon)
histogram(sizes, leg = false, color = :red, grid = false, linecolor = :white)





plotly()

# Plots the distribution of set sizes for δ in inductive_conformal function
histogram(set_sizes, color = :red, leg = false, grid = false, linecolor = :white)

# Plots the distribution of nonconformity scores
histogram(α_i, color = :red, leg = false, grid = false, linecolor = :white)

# Plots the empirical vs proposed validity
plot(confs, valid, label = "Empirical")
plot!(confs, confs, label = "Proposed")
