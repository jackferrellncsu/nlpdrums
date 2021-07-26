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
# ------------------------ Data Cleaning ------------------------ #

include("brown_functions.jl")

# Reading in text file
brown_df = CSV.read("brown.csv", DataFrame)
brown_data = brown_df[!, 4]
raw_sentences = split.(brown_data, " ")

# Finding unique words and embeddings for each
unique_pos, sentences, sentence_tags = data_cleaner(raw_sentences)
words = get_word_vec(sentences)
unique_words = convert(Vector{String},unique(words))

# Finding embeddings for each unique word
#embeddings_glove = load_embeddings(GloVe{:en},4, keep_words=Set(unique_words))
#embtable = Dict(word=>embeddings_glove.embeddings[:,ii] for (ii,word) in enumerate(embeddings_glove.vocab))
embtable = JLD.load("brownEmbs.jld", "embtable")
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



temp_train, test, temp_train_class, test_class = SampleMats(sent_tens_emb, onehot_vecs) |> gpu
train, calib, train_class, calib_class = SampleMats(temp_train, temp_train_class) |> gpu

println("Phase 2 complete")

# Creating DataLoader
dl_calib = Flux.Data.DataLoader((calib, calib_class))
dl_test = Flux.Data.DataLoader((test, test_class))
dl_train = Flux.Data.DataLoader((train, train_class),
                                    batchsize = 100, shuffle = true)


#data = collect(zip(train, train_class))

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

vectorizer(x) = embedding(BLSTM(x)) |> gpu

model(x) = predictor(vectorizer(x)) |> gpu


# Optimizer
opt = RADAM()

# Parameters
ps = Flux.params((forward, backward, embedding, predictor))

# Loss
function loss(x, y)
    Flux.reset!(forward)
    Flux.reset!(backward)
    l = Flux.crossentropy(model(x), y)
    return l
end

# Training the Neural Net, Tracking Loss Progression
epochs = 5
traceY = []
println("Beginning training")

#evalcb() = push!(traceY, loss(train[:, :, 100]), train_class[:, 100]))

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

println("training complete")
model = model |> cpu
using BSON: @save

BSON.@save "lstm_mod.bson" model

JLD.save("trace.jld", "trace", traceY)


# # Plots Loss
# plotly()
# x = 1:epochs
# y = traceY
# plot(x, y)
