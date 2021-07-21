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

# Masks random word in each sentence
masked_word, masked_pos, new_sentences = word_masker(sentences, sentence_tags)

# mask_ind - the index of the masked word in each sentence
# mask_emb - the embedding of each masked word
# sent_emb - the embeddings of each word in every sentence
# pre_sent_embs - the embeddings of the sentence up to the masked word
# post_sent_embs - the embeddings of the sentence after the masked word
mask_ind, mask_emb, sent_emb, pre_sent_embs,
                    post_sent_embs = create_embeddings(masked_word,
                    masked_pos, new_sentences, embtable)

sent_emb = convert(Vector{Vector{Vector{Float32}}}, sent_emb)

onehot_vecs = []
onehot_vecs = convert(Vector{Vector{Float32}}, onehot_vecs)

for i in 1:length(masked_pos)
    push!(onehot_vecs, Flux.onehot(masked_pos[i], unique_pos))
end

train, train_class, test, test_class, calib,
                calib_class = splitter(sent_emb, onehot_vecs, .9, .9)

dl_calib = Flux.Data.DataLoader((calib, calib_class))
dl_test = Flux.Data.DataLoader((test, test_class))
dl_train = Flux.Data.DataLoader((train[1:10], train_class[1:10]),
                                    batchsize = 10000, shuffle = true)


forward = LSTM(300, 150)
backward = LSTM(300, 150)
embedding = Dense(300, 300)
predictor = Chain(Dense(300, 250, relu), Dense(250,190), softmax)

BLSTM(x) = vcat(forward.(x)[end], backward.(reverse(x))[end])

vectorizer(x) = embedding(BLSTM(x))

model(x) = predictor(vectorizer(x))

# Optimizer
opt = RADAM()

# Parameters
ps = Flux.params((forward, backward, embedding, predictor))

# Loss
function loss(x, y)
    Flux.reset!((forward, backward))
    return sum(Flux.Losses.crossentropy.(model.(x), y))
end


# Training the Neural Net, Tracking Loss Progression
epochs = 1
traceY = []
for i in 1:epochs
    println("Starting epoch #", i, " ...")
    Flux.train!(loss, ps, dl_train, opt)
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
