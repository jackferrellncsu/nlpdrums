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

# ------------------------ Data Cleaning ------------------------ #

# Reading in text file
og_text = open("/Users/eplanch/Downloads/1342-0.txt", "r")
corpus = read(og_text, String)
close(og_text)

pride_jld_creator(corpus)

# Loads the PridePrej JLD file in
pride_jld = JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/PridePrej.jld")
embedding_table = pride_jld["embtable"]
word_index = pride_jld["word_index"]
embtable_back = pride_jld["embtable_back"]
corp = pride_jld["corpus"]
split_sent = pride_jld["split_sentences"]
pre_sentence = pride_jld["data"][:, 1]
next_word = pride_jld["data"][:, 2]
data = pride_jld["data"]

# Removing words with no embedding
unique_words = [word for word in keys(embtable)]
data = DataFrame(data)
filter!(row -> row[2] ∈ unique_words, data)
data = Matrix(data)

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

# Counting how many zeros are in the nextword_emb
# no need to run again, curiosity
counter = 0
for i in 1:length(nextword_emb[1, :])
    if nextword_emb[:, i] == zeros(300)
        counter += 1
    end
end

#ignore
for i in 1:counter
    for j in 1:length(nextword_emb[1, :])
        if nextword_emb[:, j] == zeros(300)
            nextword_emb = drop_rc(nextword_emb, c=j)
            break
        end
    end
end

# Creating the tensor for the pre-sentence embeddings
tensor = create_tensor(embedding_table, pre_sentence, 300, 5)

#ignore
for i in 1:counter
    for j in 1:length(nextword_emb[1, :])
        if nextword_emb[:, j] == zeros(300)
            pre_sentence = drop_rc(pre_sentence, c=j)
            break
        end
    end
end

# Splitting tensor into train/test/calib
train_tens_raw, test_tens_raw, calib_tens_raw = split_tensor(tensor, nextword_emb, .9, .9)

# Matrices for classification
train_tens_class = train_tens_raw[:, end, :]
test_tens_class = test_tens_raw[:, end, :]
calib_tens_class = calib_tens_raw[:, end, :]

# Tensors for convolution
train_tens = train_tens_raw[:, 1:5, :]
test_tens = test_tens_raw[:, 1:5, :]
calib_tens = calib_tens_raw[:, 1:5, :]

# Convolution
conv_train = convolute_channel(train_tens, 4, relu)
conv_test = convolute_channel(test_tens, 4, relu)
conv_calib = convolute_channel(calib_tens, 4, relu)

# Creation of DataLoader objects
dl_calib = Flux.Data.DataLoader((conv_calib, calib_tens_class))
dl_test = Flux.Data.DataLoader((conv_test, test_tens_class))
dl_train = Flux.Data.DataLoader((conv_train, train_tens_class),
                                    batchsize = 100, shuffle = true)

# ----------------------- Neural Net ----------------------- #

# Neural Net Architecture
nn = Chain(
    Dense(1200, 1100, relu),
    Dense(1100, 1000, relu),
    Dense(1000, 900, x->x)
    )

# Optimizer
opt = RADAM()

# Parameters
ps = Flux.params(nn)

# Loss Function
function loss(x, y)
    N = nn(x)
    return minimum([norm(N[1:300] .- y), norm(N[301:600] .- y), norm(N[601:900] .- y)])
end

# Training the Neural Net, Tracking Loss Progression
totalLoss = []
traceY = []
for i in ProgressBar(1:50)
    Flux.train!(loss, ps, dl_train, opt)
    totalLoss = 0
    for (x,y) in dl_train
        totalLoss += loss(x,y)
    end
    push!(traceY, totalLoss)
end

# Saving Model
using BSON: @save
@save "emi_conf_multivector3.bson" nn

# ----------------------- Conformal ----------------------- #

using BSON: @load
BSON.@load "emi_conf_multivector3.bson" nn

# Conformal predictions
setto = Vector{String}()
setto = inductive_conformal(nn, 0.05, dl_test)

# ------------------------ Testing ------------------------ #

# Checking accuracy
acc = 0.0
for (x,y) in dl_calib
    acc += (norm(y - find_prediction(x, y, nn))).^2
end
mse_acc = acc / length(conv_calib[1,:])


# Trace Plot 1 (loss vs epochs)
x = 1:50
y = traceY
plotly()
plot(x, y, label = "Loss Progression")
xlabel!("Total # of epochs")
ylabel!("Loss values")

# Finds index of set with the least number of predictions (> 0)
minu = 0
for i in 1:length(setto)
    if length(setto[i]) > 0
        if i == 1
            minu = i
        elseif length(setto[i]) < length(setto[minu])
            minu = i
        end
    end
end

# Visualization of how the model did, shows pre-sentence, real next word
# and predicted next word
act_sent = []
sent_index = minu
for i in 1:length(test_tens_raw[1, :, 1])
    push!(act_sent, get(embtable_back, test_tens_raw[:, i, sent_index], 0))
end
real_sent = act_sent[5] * act_sent[4] * act_sent[3] * act_sent[2] * act_sent[1]
real_word = act_sent[6]
println("REAL SENTENCE : " * real_sent)
println("REAL NEXTWORD : " * real_word)
print("PREDICTED NEXTWORDS : ")
println(setto[sent_index])

# Calculating the percentage of how many of the next words fall into the
# their corresponding prediction region
region_accuracy = conf_accuracy(embtable_back, setto, test_tens_raw, test_tens_class, dl_test)
