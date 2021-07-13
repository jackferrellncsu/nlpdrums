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

# ------------------------ Functions ------------------------ #

# Cleans data by removing puctuation, making text lowercase and
# making sentences seperable by ".". Also removes chapter labels
# @param corp - corpus of text that needs to be cleaned
# @return corp - cleaned data
function data_cleaner(corp)

    # Removing "\r" and "\n"
    corp = replace(corp, "\r" => "")
    corp = replace(corp, "\n" => "")

    # Removing the chapter labels
    for i in 1:100
        corp = replace(corp, "chapter "*string(i) => "")
    end

    # Removing punctuation
    corp = replace(corp, "”" => "")
    corp = replace(corp, "“" => "")
    corp = replace(corp, ";" => "")
    corp = replace(corp, ":" => "")
    corp = lowercase(corp)
    corp = replace(corp, "mr." => "mr")
    corp = replace(corp, "mrs." => "mrs")
    corp = replace(corp, "dr." => "dr")
    corp = replace(corp, "." => " .")
    corp = replace(corp, "!" => " .")
    corp = replace(corp, "?" => " .")
    corp = replace(corp, "," => "")
    corp = replace(corp, "_" => "")
    corp = replace(corp, "(" => "")
    corp = replace(corp, ")" => "")
    corp = replace(corp, "-" => "")
    corp = replace(corp, "—" => " ")

    return corp
end

# Creates a vector of vectors of individual words in each sentence
# @param sentences - vector of sentences
# @return splits - vector of vectors of individual words in each sentence
function split_sentences(sentences)

    splits = []
    for i in 1:length(sentences)
        sent = convert(Vector{String}, split(sentences[i], " "))
        push!(splits, append!(sent, [String(".")]))
        splits[i] = filter(x -> x ≠ "", splits[i])
    end

    # Reomving periods
    for i in splits
        filter!(x -> x != ".", i)
    end

    return splits
end

# Creates a tensor of word embeddings for any number of sentences
# @param embeddings - embedding dictionary for words
# @param sentences - all possible pre-sentences (vector of vectors)
    # all sentences must be in order from last word to first word
    # eg. "it is a dog" -> ["dog", "a", "is", "it"]
# @param num_embed - length of embedding vectors
# @param num_words - max number of words considered in a sentence
# @return tens - tensor of word embeddings for any number of sentences
function create_tensor(embeddings, sentences, num_embed, num_words)

    # Creates tensor of proper dimensions, fills with zeros
    tens = zeros(num_embed, num_words, length(sentences))

    # Loops through every possible pre-sentence
    for i in 1:length(sentences)

        vec_sum = zeros(num_embed)

        # Loops through each word (String) in each pre-sentence
        for j in 1:length(sentences[i])

            if length(sentences[i]) < num_words

                # Fills tensor with embeddings of last "num_words" words in each pre-sentence
                tens[:, j, i] = get(embeddings, sentences[i][j], zeros(num_embed))

                # Sums up to last "num_words" words in each pre-sentence
                vec_sum += get(embeddings, sentences[i][j], zeros(num_embed))

            elseif length(sentences[i]) >= num_words

                # Stops loop once last "num_words" words are added to tensor
                if j == (num_words + 1)
                    break
                else
                    tens[:, j, i] = get(embeddings, sentences[i][j], zeros(num_embed))
                end
            end
        end

        # For pre-sentences with < "num_words", average of word embeddings
        # is taken and added to tensor
        if length(sentences[i]) < num_words

            vec_avg = zeros(num_embed)
            vec_avg = vec_sum/length(sentences[i])
            vecs_needed = 0
            vecs_needed = num_words - length(sentences[i])

            for j in 1:vecs_needed
                tens[:, length(sentences[i]) + j, i] = vec_avg
            end
        end
        println(i)
    end

    return tens
end

# Splits tensor into test/train/validation sets
# @param tensor - tensor of word embeddings
# @param next_word - matrix of next word embeddings
# @param train_test - percentage of data wanted to be in train set vs test set
# @param train_calib - percentage of data wanted to be in train set vs calib set
# @return train_tens - tensor of training data
# @return test_tens - tensor of testing data
# @return calib_tens - tensor of calibration data
function split_tensor(tensor, next_word, train_test, train_calib)

    # Concatenating next word onto tensor to ensure splits don't misalign the values
    tens = zeros(length(tensor[:,1,1]), length(tensor[1,:,1]) + 1, length(tensor[1,1,:]))
    tens[1:length(tensor[:,1,1]), 1:length(tensor[1,:,1]), 1:length(tensor[1,1,:])] = tensor
    tens[1:length(tensor[:,1,1]), length(tens[1,:,1]), 1:length(tensor[1,1,:])] = next_word

    # Computing sizes of each set
    first_train_size = Int(ceil(length(tensor[1,1,:]) * train_test))
    test_size = Int(length(tensor[1,1,:]) - first_train_size)
    train_size = Int(ceil(first_train_size * train_calib))
    calib_size = Int(first_train_size - train_size)

    # Test set creation
    test_cols = sample(1:length(tensor[1,1,:]), test_size, replace = false)
    test_tens = zeros(length(tensor[:,1,1]), length(tensor[1,:,1]) + 1, test_size)

    for i in 1:length(test_cols)
        test_tens[:, :, i] = tens[:, :, test_cols[i]]
    end

    num_array = []

    for i in 1:length(tensor[1, 1, :])
        push!(num_array, i)
    end

    new_indeces = []
    new_indeces = setdiff(num_array, test_cols)

    train_cols = sample(new_indeces, train_size, replace = false)
    train_tens = zeros(length(tensor[:,1,1]), length(tensor[1,:,1]) + 1, train_size)

    for i in 1:length(train_cols)
        train_tens[:, :, i] = tens[:, :, train_cols[i]]
    end

    new_indeces = setdiff(new_indeces, train_cols)

    calib_cols = sample(new_indeces, calib_size, replace = false)
    calib_tens = zeros(length(tensor[:,1,1]), length(tensor[1,:,1]) + 1, calib_size)

    for i in 1:length(calib_cols)
        calib_tens[:, :, i] = tens[:, :, calib_cols[i]]
    end

    return train_tens, test_tens, calib_tens
end

# Convolutes singular dimension of any given tensor that is in the
# form of a x b x c where b is the dimension that needs convolution.
# Takes the number of output channels desired and vertically concatenates
# each channel to create a matrix.
    # eg. [10 x 50 x 100] -> [10 x 4 x 100] -> 40 x 100 matrix
# @param tensor - tensor of word embeddings
# @param out_ch - the number of output channels desired
# @param activation - activation function for convolutional layer
function convolute_channel(tensor, out_ch, activation)

    input_ch = length(tensor[1, :, 1])
    conv_layer = Conv(tuple(1, 1), input_ch => out_ch, activation)

    val1 = length(tensor[1, 1, :])
    val2 = length(tensor[1, :, 1])
    val3 = length(tensor[:, 1, 1])

    tens_array = reshape(tensor, (val1, val3, val2, 1))
    conv_array = conv_layer(tens_array)

    conv_mat = conv_array[:, :, 1]'
    for i in 2:out_ch
        conv_mat = vcat(conv_mat, conv_array[:, :, i]')
    end

    return conv_mat
end

# Given a model, error (δ), testing set, and nonconformity
# function, function produces a prediction region for each pres-entence
# in testing set. Number of words in each prediction region is a
# reflection of confidence (1 - δ).
# @param model - trained model used to predict  next words
# @param δ - error, used to compute confidence (1 - δ)
# @param dl_test - testing set (may be in the form of a dataloader)
# @param nonconf - nonconformity function
# @return collection - a set of prediction regions for the next word
# of each pre-sentence
function inductive_conformal(model, δ, dl_test)

    confidence = 1 - δ

    α_i = Vector{Float64}()
    for (x, y) in dl_test
        α = norm(y - find_prediction(x, y, model))
        push!(α_i, α)
        println(length(α_i)/length(dl_test))
    end

    sort!(α_i, rev = true)

    α_k = 0
    collection = Vector{Vector{String}}()
    for (x, y) in dl_test
        region = Vector{String}()
        global α_k = norm(y - find_prediction(x, y, model))
        push!(α_i, α_k)
        quant = quantile(α_i, confidence)
        pred = find_prediction(x, y, model)
        for i in embedding_table
            distance = norm(pred - i[2])
            if distance <= quant
                push!(region, i[1])
            end
        end
        push!(collection, region)
        println(length(collection))
        α_i = α_i[1:end-1]
    end

    return collection
end

# Not a generalizable function; used to find the best neural net
# prediction (or the closest prediction to the actual value)
function find_prediction(x, y, N)
    pred1 = norm(N(x)[1:300] .- y)
    pred2 = norm(N(x)[301:600] .- y)
    min = minimum([pred1, pred2])
    if min == pred1
        act_pred = N(x)[1:300]
    elseif min == pred2
        act_pred = N(x)[301:600]
    end
    return act_pred
end

# ------------------------ Data Cleaning ------------------------ #

# Reading in text file
og_text = open("/Users/eplanch/Downloads/1342-0.txt", "r")
corpus = read(og_text, String)
close(og_text)

# Cleaning data
corp = data_cleaner(corpus)

# Creating vector of sentences
sent_vec = convert(Vector{String},split(corp, "."))

# Splitting sentence vector into individual words per sentence
split_sent = split_sentences(sent_vec)

# Creates a vectors with each entry being an individual word in the corpus
word_count = convert(Vector{String},split(corp, " "))
word_count = filter(x->(x≠"" && x≠"."),word_count)

# Creates a dictionary with each word and its index
uni = convert(Vector{String}, unique(word_count))
D = Dict(uni .=> 1:length(uni))

# Loading in glove embeddings, creating embedding table
embtable_raw = load_embeddings(GloVe{:en},4, keep_words=Set(uni))
embtable = Dict(word=>embtable_raw.embeddings[:,ii] for (ii,word) in enumerate(embtable_raw.vocab))
embtable_back = Dict(embtable_raw.embeddings[:,ii]=>word for (ii,word) in enumerate(embtable_raw.vocab))

# Creates possible pre-sentence embeddings and
# possible next words
pre_sentences = []
nextword = []
for i in 1:length(split_sent)
    for ii in 1:length(split_sent[i])-1
        push!(pre_sentences, split_sent[i][1:ii])
        push!(nextword, split_sent[i][ii+1])
    end
end

# Creation of PridePrej JLD file
JLD.save("PridePrej.jld", "corpus", corp, "split_sentences", split_sent, "data", hcat(pre_sentences, nextword), "embtable", embtable)

# ------------------------ Functions ------------------------ #


# Loads the PridePrej JLD file in
pride_jld = JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/PridePrej.jld")
embedding_table = pride_jld["embtable"]
pre_sentence = pride_jld["data"][:, 1]
next_word = pride_jld["data"][:, 2]

# Reversing order of each pre-sentence
for i in 1:length(pre_sentence)
    pre_sentence[i] = reverse(pre_sentence[i])
end

# Creating the tensor for the pre-sentence embeddings
tensor = create_tensor(embedding_table, pre_sentence, 300, 5)

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
conv_train = convolute_channel(train_tens, 3, relu)
conv_test = convolute_channel(test_tens, 3, relu)
conv_calib = convolute_channel(calib_tens, 3, relu)

# Creation of DataLoader objects
dl_calib = Flux.Data.DataLoader((conv_calib, calib_tens_class))
dl_test = Flux.Data.DataLoader((conv_test, test_tens_class))
dl_train = Flux.Data.DataLoader((conv_train, train_tens_class),
                                    batchsize = 100, shuffle = true)

# ----------------------- Neural Net ----------------------- #

# Neural Net Architecture
nn = Chain(
    Dense(900, 800, relu),
    Dense(800, 700, relu),
    Dense(700, 600, x->x)
    )

# Optimizer
opt = RADAM()

# Parameters
ps = Flux.params(nn)

# Loss Function
function loss(x, y)
    N = nn(x)
    return minimum([norm(N[1:300] .- y), norm(N[301:600] .- y)])
end

# Training the Neural Net, Tracking Loss Progression
totalLoss = []
traceY = []
for i in ProgressBar(1:10)
    Flux.train!(loss, ps, dl_train, opt)
    totalLoss = 0
    for (x,y) in dl_train
        totalLoss += loss(x,y)
    end
    push!(traceY, totalLoss)
end

# Saving Model
using BSON: @save
@save "emi_conf_multivector.bson" nn

# ----------------------- Analysis ----------------------- #

using BSON: @load
BSON.@load "emi_conf_multivector.bson" nn

# Checking accuracy
acc = 0.0
α = 0
α_i = Vector{Float64}()
for (x,y) in dl_calib
    α = norm(y - find_prediction(x, y, nn))
    push!(α_i, α)
    println(length(α_i)/length(dl_calib))
    acc += (norm(y - find_prediction(x, y, nn))).^2
end
mse_acc = acc / length(conv_calib[1,:])


# Trace Plot 1 (loss vs epochs)
x = 1:10
y = traceY
plotly()
plot(x, y, label = "Loss Progression")
xlabel!("Total # of epochs")
ylabel!("Loss values")

# ----------------------- Conformal ----------------------- #

# Conformal predictions
setto = Vector{String}()
setto = inductive_conformal(nn, 0.10, dl_test)

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
actual_nextword = ""
acc = 0
for i in 1:length(test_tens_raw[1, 1, :])
    actual_nextword = get(embtable_back, test_tens_raw[:, 6, i], 0)
    if actual_nextword ∈ setto[i]
        acc += 1
    end
end
tot = acc/length(dl_test)
tot = tot * 100
print("Total accuracy: ")
print(round(tot, digits = 2))
print("%")
