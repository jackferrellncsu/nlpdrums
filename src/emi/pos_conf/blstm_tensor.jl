using SparseArrays
using SparseArrayKit
using JLD
using Word2Vec
using LinearAlgebra
using Statistics
using Embeddings
using Flux
using Flux: Losses
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

# Finding unique words and embeddings for each
unique_pos, sentences, sentence_tags = data_cleaner(raw_sentences)
words = get_word_vec(sentences)
unique_words = convert(Vector{String},unique(words))

# Finding embeddings for each unique word
embeddings_glove = load_embeddings(GloVe{:en},4, keep_words=Set(unique_words))
embtable = Dict(word=>embeddings_glove.embeddings[:,ii] for (ii,word) in enumerate(embeddings_glove.vocab))

# Finding the words that have GloVe embeddings
keys_embtable = get_keys(embtable)

# Finding the words that don't have GloVe embeddings
no_embeddings = setdiff(unique_words, keys_embtable)

sent_tens, new_sent, new_tags = sent_embeddings(sentences, sentence_tags, 300, 180, embtable)

new_sent = convert(Vector{Vector{String}}, new_sent)
new_tags = convert(Vector{Vector{String}}, new_tags)
sent_tens = convert(Array{Float32, 3}, sent_tens)

# Masks random word in each sentence
masked_word, masked_pos, new_sentences = word_masker(new_sent, new_tags)

# mask_ind - the index of the masked word in each sentence
# mask_emb - the embedding of each masked word (NOT NEEDED)
mask_ind, mask_emb, sent_tens_emb = create_embeddings(masked_word, masked_pos,
                                new_sentences, sent_tens, embtable)

sent_tens_emb = create_window(sent_tens_emb, 15)
sent_tens_emb = convert(Array{Float32, 3}, sent_tens_emb)

#onehot_mat = Flux.onehotbatch(masked_pos, unique_pos)
onehot_vecs = zeros(length(unique_pos), length(masked_pos))
for i in 1:length(masked_pos)
    onehot_vecs[:, i] = Flux.onehot(masked_pos[i], unique_pos)
end
onehot_vecs = convert(Array{Float32, 2}, onehot_vecs)



temp_train, test, temp_train_class, test_class = SampleMats(sent_tens_emb, onehot_vecs)
train, calib, train_class, calib_class = SampleMats(temp_train, temp_train_class)



# Creating DataLoader
dl_calib = Flux.Data.DataLoader((calib, calib_class))
dl_test = Flux.Data.DataLoader((test, test_class))
dl_train = Flux.Data.DataLoader((train, train_class),
                                    batchsize = 100, shuffle = true)


#data = collect(zip(train, train_class))

forward = LSTM(300, 150)
backward = LSTM(300, 150)
embedding = Dense(300, 300, mish)
predictor = Chain(Dense(300, 250, relu), Dense(250,190), softmax)

function BLSTM(x)

    fw = forward.([x[:, 1:15, i] for i in 1:size(x, 3)])
    @show typeof(fw)
    @show size(fw)
    fw_mat = hcat.(f[:,15] for f in fw)
    A = reshape(collect(fw_mat), (100, 1))
    @show typeof(A)
    @show size(A)
    #fw_mat = zeros(Float32, size(fw[1],  1), size(x, 3))

    # for (i, e) in enumerate(fw)
    #     fw_mat[:, i] = e[:, 15]
    #     break
    # end

    bw = backward.([x[:, end:-1:17, i] for i = size(x, 3):-1:1])
    bw_mat = hcat.(b[:,15] for b in bw)
    #bw_mat = zeros(Float32, size(bw[1],  1), size(x, 3))

    # for (i, e) in enumerate(bw)
    #     bw_mat[:, i] = e[:, 15]
    # end

    fw_mats = hcat(h for h in fw_mat)
    @show typeof(fw_mats)

    res = vcat.(fw_mat, bw_mat)

    return res
end

vectorizer(x) = embedding(BLSTM(x))

model(x) = predictor(vectorizer(x))

for (x, y) in dl_train
    @show typeof(BLSTM(x))
    @show size(BLSTM(x))
    @show size(BLSTM(x)[1])

    break
end

# Optimizer
opt = RADAM()

# Parameters
ps = Flux.params((forward, backward, embedding, predictor))

# Loss
function loss2(x, y)

    #Flux.reset!((forward, backward))

    @show "We win"
    return crossentropy(model(x), y)
end

#=
for (x, y) in dl_train
    #print(x)
    print(size(x[1][:,1]))
    #print(size(x[1]))
    #print(typeof(x))
    #print(typeof(x[1]))
    break
end
=#

# Training the Neural Net, Tracking Loss Progression
epochs = 1
traceY = []

for i in ProgressBar(1:epochs)
    Flux.train!(loss2, ps, dl_train, opt)
    Flux.reset!((forward, backward))
    L = sum(loss(sent_emb[1:100], onehot_vecs[1:100]))
    push!(traceY, L)
    for ii in length(sent_emb)
        Flux.reset!((forward, backward))
        sent_emb[ii][mask_ind[ii]] = vectorizer(sent_emb[ii])
    end
end

# Plots Loss
plotly()
x = 1:epochs
y = traceY
plot(x, y)


function predict(x)
    pred = model1(x)
    Flux.reset!((forward, backward))
    return pred
end
