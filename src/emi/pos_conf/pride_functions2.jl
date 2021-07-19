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
        α = norm(y - nn(x))
        push!(α_i, α)
        println(length(α_i)/length(dl_test))
    end

    sort!(α_i, rev = true)

    α_k = 0
    collection = Vector{Vector{String}}()
    for (x, y) in dl_test
        region = Vector{String}()
        global α_k = norm(y - nn(x))
        push!(α_i, α_k)
        quant = quantile(α_i, confidence)
        pred = nn(x)
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
    pred3 = norm(N(x)[601:900] .- y)
    min = minimum([pred1, pred2, pred3])
    if min == pred1
        act_pred = N(x)[1:300]
    elseif min == pred2
        act_pred = N(x)[301:600]
    elseif min == pred3
        act_pred = N(x)[601:900]
    end
    return act_pred
end

# Gives accuracy of conformal predictor sets
function conf_accuracy(dict, set, test_mat, test_class, dl_test)

    actual_nextword = ""
    acc = 0
    for i in 1:length(test_mat[1, 1, :])
        actual_nextword = get(dict, test_class[:, i], 0)
        if actual_nextword ∈ set[i]
            acc += 1
        end
    end
    tot = acc/length(dl_test)
    not = tot * 100
    rot = round(not, digits = 2)
    hot = print("Total accuracy: ", rot, "%")
    return hot
end

# Drops rows and columns of any given matrix
function drop_rc(x; r=nothing, c=nothing)
    nr, nc = size(x)
    if !isnothing(r)
        return x[setdiff(1:nr, r...), :]
    elseif !isnothing(c)
        return x[:, setdiff(1:nc, c...)]
    else
        return x
    end
end

# Not a generalizable function; used to create corpus for pride and prej
function pride_jld_creator(corpus)

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
    word_index = Dict(word=>ii for (ii, word) in enumerate(embtable_raw.vocab))

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
    JLD.save("PridePrej.jld", "corpus", corp, "split_sentences", split_sent, "data", hcat(pre_sentences, nextword),
                                    "embtable", embtable, "embtable_back", embtable_back, "word_index", word_index)
end

function split_tensor_onehot(tensor, next_word, train_test, train_calib)

    new_tens = zeros(6135, 5, length(tensor[1, 1, :]))
    for i in 1:length(new_tens[1, :, 1])
        println(i)
        for j in 1:length(new_tens[1, 1, :])
            new_tens[1:300, i, j] = tensor[:, i, j]
        end
    end

    tensor = new_tens

    new_class = zeros(6135, 1, length(tensor[1, 1, :]))
    for i in 1:length(new_class[1, 1, :])
        new_class[:, 1, i] = next_word[:, i]
    end

    # Concatenating next word onto tensor to ensure splits don't misalign the values
    tens = zeros(length(tensor[:,1,1]), length(tensor[1,:,1]) + 1, length(tensor[1,1,:]))
    tens[1:length(tensor[:,1,1]), 1:length(tensor[1,:,1]), 1:length(tensor[1,1,:])] = tensor
    tens[1:length(tensor[:,1,1]), length(tens[1,:,1]), 1:length(tensor[1,1,:])] = new_class

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

function data_class_split(tensor)

    data = tensor[1:300, 1:5, :]
    class = tensor[:, end, :]

    return data, mat2vec(class)
end

function mat2vec(matrix)

    vec = []
    for i in 1:length(matrix[1,:])
        push!(vec, matrix[:, i])
    end

    return vec
end

function get_onehot_tensor(tensor, y_class)

    one_hots = BitArray(undef, 6135, length(tensor[1, end, :]))
    for i in 1:length(tensor[1, end, :])
        word = get(embtable_back, tensor[:, end, i], 0)
        for j in 1:length(next_word)
            if word == next_word[j]
                one_hots[:,i] .= y_class[:,j]
            end
        end
        println(i)
    end

    return one_hots
end

function create_class(matrix, next_word, length, unique_words)

    next_word = next_word[2:end]
    class = BitArray(undef, length, length(next_word[1:length(matrix[1,:])]))
    for i in 1:length(matrix[1,:])
        class[:, i]  = (Flux.onehot(next_word[i], unique_words) .== 1)
    end
    return class
end

function split_classes(matrix, next_word, length, train_test, train_calib, unique_words)

    # Computing sizes of each set
    first_train_size = Int(ceil(length(matrix[1,:]) .* train_test))
    test_size = Int(length(matrix[1,:]) - first_train_size)
    train_size = Int(ceil(first_train_size * train_calib))
    calib_size = Int(first_train_size - train_size)

    train = matrix[:, 1:train_size]
    train_class = create_class(train, next_word, 6135, unique_words)

    test = matrix[:, train_size+1:train_size+test_size]
    test_class = create_class(test, next_word, 6135, unique_words)

    calib = matrix[:, test_size+train_size+1:train_size+test_size+calib_size]
    calib_class = create_class(calib, next_word, 6135, unique_words)

    return train, train_class, test, test_class, calib, calib_class
end
