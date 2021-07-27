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

include("brown_functions.jl")

Random.seed!(24)

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

# Splitting the data and one hot matrix into train/test/calib
temp_train, test, temp_train_class, test_class = SampleMats(sent_tens_emb, onehot_vecs)
train, calib, train_class, calib_class = SampleMats(temp_train, temp_train_class)

# Creating DataLoader
dl_calib = Flux.Data.DataLoader((calib, calib_class))
dl_test = Flux.Data.DataLoader((test, test_class))
dl_train = Flux.Data.DataLoader((train, train_class),
                                    batchsize = 100, shuffle = true)

# Neural Network Architecture
forward = LSTM(300, 150)
backward = LSTM(300, 150)
embedding = Dense(300, 300)
predictor = Chain(Flux.Dropout(.2; dims=1), Dense(300, 250, relu), Dense(250,190), softmax)

# BLSTM Layer
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

# Predicts word embedding for masked word
vectorizer(x) = embedding(BLSTM(x))

# Predicts part of seech for predicted word embedding
model(x) = predictor(vectorizer(x))

# Optimizer
opt = RADAM()

# Parameters
ps = Flux.params((forward, backward, embedding, predictor))

# Loss
function loss(x, y)
    l = sum(Flux.Losses.crossentropy(model(x), y))
    Flux.reset!((forward, backward))
    return l
end

# Training the Neural Net, Tracking Loss Progression
epochs = 1
traceY = []

# Training loop for neural net
for i in ProgressBar(1:epochs)
    Flux.train!(loss, ps, dl_train, opt)
    #Flux.reset!((forward, backward))
end

using BSON: @load
BSON.@load "lstm_mod_1.bson" model

percent_right, set_sizes, sets = inductive_conformal(model, .10, dl_calib, dl_test, unique_pos)
tag_freq("ppss+bbb*")

#δ = .10
#confidence = 1 - δ

#α_i = Vector{Float64}()
#for (x, y) in dl_calib
#    cor = maximum(y .* model(x))
#    α = 1 - cor
#    push!(α_i, α)
#    println(length(α_i)/length(dl_calib))
#end

#α_k = 0
#eff = []
#correct = []
#sets = []
#Q = quantile(α_i, confidence)
#for (x, y) in dl_test
#    global α_k = 1 .- model(x)
#    p_k = α_k .<= Q
#    push!(eff, sum(p_k))
#    push!(correct, p_k[argmax(y)] == 1)
#    temp = []
#    for j in 1:length(p_k)
#        if p_k[j] == 1
#            push!(temp, unique_pos[j])
#        end
#    end
#    push!(sets, temp)
#end


#d = Vector{Float64}()
#for (x, y) in dl_calib
#    cor = maximum(y .* lstm6(x))
#    α = 1 - cor
#    push!(d, α)
#    println(length(d)/length(dl_calib))
#end

#α_k = 0
#eff = []
#correct = []
#sets = []
#Q = quantile(d, confidence)
#for (x, y) in dl_test
#    global α_k = 1 .- lstm6(x)
#    p_k = α_k .<= Q
#    push!(eff, sum(p_k))
#    push!(correct, p_k[argmax(y)] == 1)
#    temp = []
#    for j in 1:length(p_k)
#        if p_k[j] == 1
#            push!(temp, unique_pos[j])
#        end
#    end
#    push!(sets, temp)
#end


sets = convert(Vector{Vector{String}}, sets)
perc = convert(Float32, mean(correct))
eff = convert(Vector{Int64}, eff)
