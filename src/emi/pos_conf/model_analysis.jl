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

using BSON: @load
BSON.@load "lstm_param.bson" weights


forward = LSTM(300, 150) |> gpu
backward = LSTM(300, 150) |> gpu
embedding = Dense(300, 300)|> gpu
predictor = Chain(Dense(300, 250, relu), Dense(250,190), softmax)|> gpu

function BLSTM(x)

    #Flux.reset!((forward, backward))
    fw = forward.([x[:, 1:15, i] for i in 1:size(x, 3)])
    fw_mat = hcat.(f[:,15] for f in fw)

    bw = backward.([x[:, end:-1:17, i] for i = size(x, 3):-1:1])
    bw_mat = hcat.(b[:,15] for b in bw)

    fw_temp = fw_mat[1]
    for i in 2:length(fw_mat)
        fw_temp = hcat(fw_temp, fw_mat[i])
    end

    bw_temp = bw_mat[1]
    for i in 2:length(bw_mat)
        bw_temp = hcat(bw_temp, bw_mat[i])
    end
    #@show fw_temp
    res = vcat(fw_temp, bw_temp)
    #@show res
    return res
end

vectorizer(x) = embedding(BLSTM(x))

model(x) = predictor(vectorizer(x))

loaded_model = Flux.loadparams!(model, weights)



# Given a model, delta (δ), testing set, and nonconformity
# function, function produces a prediction region for each pre-sentence
# in testing set. Number of words in each prediction region is a
# reflection of confidence (1 - δ).
# @param model - trained model used to predict  next words
# @param δ - error, used to compute confidence (1 - δ)
# @param dl_test - testing set (may be in the form of a dataloader)
# @param nonconf - nonconformity function
# @return collection - a set of prediction regions for the next word
# of each pre-sentence
function inductive_conformal(mod, δ, dl_calib, dl_test, unique_pos)

    confidence = 1 - δ

    α_i = Vector{Float64}()
    α = 0.0
    cor = 0.0
    for (x, y) in dl_calib
        cor = maximum(y .* mod(x))
        α = 1 - cor
        push!(α_i, α)
        println(length(α_i)/length(dl_calib))
    end

    sort!(α_i, rev = true)

    α_k = 0
    eff = []
    correct = []
    sets = []
    Q = quantile(α_i, confidence)
    for (x, y) in dl_test
        α_k = 1 .- mod(x)
        p_k = α_k .<= Q
        push!(eff, sum(p_k))
        push!(correct, p_k[argmax(y)] == 1)
        temp = []
        for j in 1:length(p_k)
            if p_k[j] == 1
                push!(temp, unique_pos[j])
            end
        end
        push!(sets, temp)
    end

    sets = convert(Vector{Vector{String}}, sets)
    perc = convert(Float32, mean(correct))
    eff = convert(Vector{Int64}, eff)

    return perc, eff, sets
end

percent_right, set_sizes, sets = inductive_conformal(model, .10, dl_calib, dl_test, unique_pos)
