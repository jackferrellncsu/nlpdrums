using JLD
using LinearAlgebra
using Statistics
using Embeddings
using Flux
using Random
using DataFrames
using Plots
using StatsBase
using BSON

# -------------------------------- Functions -------------------------------- #

"""
    word_masker(sentences, tags)

Mask a randomly selected word in every sentence.

Parameter sentences (Vector{Vector{String}}) - Vector of all sentences where
    each word in each sentence vector is stored as a string.
Parameter tags (Vector{Vector{String}}) - Vector of all tags for each word in
    every sentence. Corresponds with the sentences vector of vectors.

Return act_word (Vector{Any}) - Vector of words that is masked in each sentence.
Return act_pos (Vector{Any}) - Vector of masked words corresponding parts of
    speech.
Return sentences (Vector{Vector{String}}) - New sentences vector with each
    masked word being replaced with the string "/MASK/".
"""
function word_masker(sentences, tags)

    act_word = []
    act_pos = []
    for i in 1:length(sentences)
        samp = sample(1:length(sentences[i]))
        push!(act_word, sentences[i][samp])
        push!(act_pos, tags[i][samp])
        sentences[i][samp] = "/MASK/"
    end
    return act_word, act_pos, sentences
end

"""
    data_cleaner(raw_sentences)

Clean Brown Corpus by removing the tags for titles, headings, foreign words and
emphasized words. Reduce number of unique tags to 190.

Parameter raw_sentences ((Vector{Vector{SubString{String}}})) - Vector of all
    sentences where each word/tag combination in each sentence vector is stored
    as a string.

Return tagger (Vector{Any}) - Vector of unique parts of speech stored as strings
Return raw_words (Vector{Any}) - Vector of all sentences where each word in each
    sentence vector is stored as a string.
Return raw_tags (Vector{Any}) - Vector of all tags for each word in every
    sentence. Corresponds with the raw_words vector.
"""
function data_cleaner(raw_sentences)

    raw_tags = []
    raw_words = []
    for sent in raw_sentences
        raw_tags_temp = []
        raw_words_temp = []
        for word in sent
            ind = findlast(x -> x == '/', word)
            pos = word[ind+1:end]
            pos = replace(pos, "-tl" => "")
            pos = replace(pos, "-hl" => "")
            pos = replace(pos, "fw-" => "")
            pos = replace(pos, "-nc" => "")
            pos = replace(pos, "bez" => "bbb")
            push!(raw_tags_temp, convert(String, pos))
            push!(raw_words_temp, lowercase(word[1:ind-1]))
        end
        push!(raw_tags, raw_tags_temp)
        push!(raw_words, raw_words_temp)
    end

    individual_tags = []
    for i in 1:length(raw_tags)
        for j in 1:length(raw_tags[i])
            push!(individual_tags, (raw_tags[i][j]))
        end
    end

    tagger = unique!(individual_tags)

    return tagger, raw_words, raw_tags
end

"""
    get_word_vec(sentences)

Create a vector filled with every element (word) in sentences.

Parameter sentences (Vector{Any}) - Vector of all sentences where
    each word in each sentence vector is stored as a string.

Return words (Vector{String}) - Vector of every word in sentences, stored as
    strings.
"""
function get_word_vec(sentences)

    words = Vector{String}()
    for i in 1:length(sentences)
        for j in 1:length(sentences[i])
            push!(words, sentences[i][j])
        end
    end
    return words
end

"""
    sent_embeddings(sentences, sentence_tags, num_embed, num_words, dict)

Create a tensor of word embeddings (num_embed x num_words x length(sentences)).
Truncates and pads with zeros accordingly.

Parameter sentences (Vector{Any}) - Vector of all sentences where
    each word in each sentence vector is stored as a string.
Parameter sentence_tags (Vector{Any}) - Vector of all tags for each word in
    every sentence. Corresponds with the sentences vector.
Parameter num_embed {Int64} - Length of embedding vectors.
Parameter num_words {Int64} - Number of spaces every sentence will take up.
Parameter dict (Dict{String, Vector{Float32}}) - Embedding dictionary that has
    each word as a key and each words embedding as a value.

Return tens (Array{Float64, 3}) - Tensor of word embeddings for each word in
    each sentence (num_embed x num_words x length(sentences)).
Return sent (Vector{Any}) - Vector of possibly truncated sentences, depending on
    the num_words value. If num_words is greater than the length of the longest
    sentence, then sent will just be the same as the parameter sentences.
Return tags (Vector{Any}) - Vector of tags for each word in each sentence,
    corresponds with the sent vector of vectors.
"""
function sent_embeddings(sentences, sentence_tags, num_embed, num_words, dict)

    # Embeddings for each sentence
    tens = zeros(num_embed, num_words, length(sentences))

    sent = []
    tags = []
    for i in 1:length(sentences)
        temp = []
        temp_tag = []
        for j in 1:length(sentences[i])
            if length(sentences[i]) < num_words
                tens[:, j, i] = get(dict, sentences[i][j], zeros(num_embed))
                push!(temp, sentences[i][j])
                push!(temp_tag, sentence_tags[i][j])
            elseif length(sentences[i]) >= num_words
                if j == (num_words + 1)
                    break
                else
                    tens[:, j, i] = get(dict, sentences[i][j], zeros(num_embed))
                    push!(temp, sentences[i][j])
                    push!(temp_tag, sentence_tags[i][j])
                end
            end
        end
        # Pre-sentences with < "num_words" are padded with zeros
        if length(sentences[i]) < num_words

            vecs_needed = 0
            vecs_needed = num_words - length(sentences[i])

            for j in 1:vecs_needed
                tens[:, length(sentences[i]) + j, i] .= 0.0
            end
        end
        push!(sent, temp)
        push!(tags, temp_tag)
    end
    return tens, sent, tags
end

"""
    masked_embeddings(new_sentences, sent_tens)

Find indices for the masked word in each sentence. Go through embedded tensor
and replace the masked embedding with a num_embed length vector filled with
Float64 values -20.0.

Parameter new_sentences (Vector{Vector{String}}) - Sentence vector with the
    masked word in each sentence being replaced with the string "/MASK/".
Parameter sent_tens (Array{Float64, 3}) - Tensor of word embeddings for each
    word in each sentence.
Parameter num_embed {Int64} - Length of embedding vectors.

Return masked_ind (Vector{Any}) - Vector of the indices for masked word in each
    sentence.
Return sent_tens (Array{Float64, 3}) - Tensor of word embeddings for each word
    in each sentence (num_embed x num_words x length(sentences)) with each
    masked embedding being replaced with a vector of -20.0's.
"""
function masked_embeddings(new_sentences, sent_tens, num_embed)

    # Finds indices of masked words for each sentence
    masked_ind = []
    for i in 1:length(new_sentences)
        for j in 1:length(new_sentences[i])
            if new_sentences[i][j] == "/MASK/"
                push!(masked_ind, j)
            end
        end
    end

    # Embeddings for each sentence
    for i in 1:length(sent_tens[1, 1, :])
        temp = []
        mask = masked_ind[i]
        sent_tens[:, mask, i] = fill(-20.0, num_embed)
    end


    return masked_ind, sent_tens
end

"""
    create_window(sent_tens_emb, window_size)

Modify an embedding tensor by placing the masked word in the center of each
"sentence" and truncating/padding on either side of the word to meet the
window size.

Parameter sent_tens_emb (Array{Float64, 3}) - Tensor of word embeddings for each
    word in each sentence with each masked embedding being replaced with a
    vector of -20.0's.
Parameter window_size {Int64} - Number of words of either side of the masked
    word.

Return new_tens (Array{Float64, 3}) - Tensor of word embeddings for each word in
    each sentence with each masked embedding being in the center of each
    sentence.
"""
function create_window(sent_tens_emb, window_size)

    mask_ind = 0
    num_wind = (window_size * 2) + 1
    new_tens = zeros(300, num_wind, length(sent_tens_emb[1, 1, :]))
    for i in 1:length(sent_tens_emb[1, 1, :])
        mask_ind = argmin(sent_tens_emb[1, :, i])
        if mask_ind == 16
            new_tens[:, :, i] = sent_tens_emb[:, 1:num_wind, i]
        elseif mask_ind > 16
            num_start = (mask_ind - 16) + 1
            num_end = num_wind + (num_start - 1)
            new_tens[:, :, i] = sent_tens_emb[:, num_start:num_end, i]
        elseif mask_ind < 16
            num_zero = window_size - (mask_ind - 1)
            new_mat = zeros(300, num_zero)
            stopper = mask_ind + (window_size)
            new_tens[:, :, i] = hcat(new_mat, sent_tens_emb[:, 1:stopper, i])
        end
    end
    return new_tens
end

"""
    SampleMats(x_mat, y_mat, prop = 0.9)

Split data into training, training class, test, and test class for model input.

Parameter x_mat (Array{Float32, 3}) - Tensor of word embeddings for each
    word in each sentence with each masked embedding being replaced with a
    vector of -20.0's.
Parameter y_mat (Matrix{Float32}) - Matrix of one hot vectors corresponding to
    the tensor of word embeddings.

Return train_x (Array{Float32, 3}) - Training tensor for model.
Return test_x  (Array{Float32, 3}) - Testing tensor for model.
Return train_y (Matrix{Float32}) - Training class for model.
Return test_y (Matrix{Float32}) - Testing class for model.
"""
function SampleMats(x_mat, y_mat, prop = 0.9)

    inds = [1:size(x_mat)[3];]
    length(inds)
    trains = sample(inds, Int(floor(length(inds) * prop)), replace = false)
    inds = Set(inds)
    trains = Set(trains)
    tests = setdiff(inds, trains)

    train_x = x_mat[:, :, collect(trains)]
    train_y = y_mat[:, collect(trains)]

    test_x = x_mat[:, :, collect(tests)]
    test_y = y_mat[:, collect(tests)]


    return train_x, test_x, train_y, test_y
end

"""
    inductive_conformal(mod, ??, dl_calib, dl_test, unique_pos)

Compute the conformal prediciton sets for the given trained model.

Parameter mod (trained model) - Trained model for set prediction.
Parameter ?? (Int64) - epsilon value, used to compute confidence (1 - ??).
Parameter dl_calib (Dataloader) - calibration dataloader.
Parameter dl_test (Dataloader) - testing dataloader.
Parameter unique_pos (Vector{Any}) - Unique parts of speech in corpus.

Return perc (Float32) - proportion of the amount of predicitons the set got
    right.
Return eff (Vector{Int64}) - vector of the length value for each set.
Return sets (Vector{Vector{String}}) - Vector of vector contataing the
    prediction sets.
"""
function inductive_conformal(mod, ??, dl_calib, dl_test, unique_pos)

    confidence = 1 - ??

    ??_i = Vector{Float64}()
    ?? = 0.0
    cor = 0.0
    for (x, y) in dl_calib
        cor = maximum(y .* mod(x))
        ?? = 1 - cor
        push!(??_i, ??)
        println(length(??_i)/length(dl_calib))
    end

    sort!(??_i, rev = true)

    ??_k = 0
    eff = []
    correct = []
    sets = []
    Q = quantile(??_i, confidence)
    for (x, y) in dl_test
        ??_k = 1 .- mod(x)
        p_k = ??_k .<= Q
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

    return perc, correct, eff, sets
end

"""
    tag_freq(tag)

Testing method that counts how many times any given tag appears in the
prediction sets.

Parameter tag (String) - Tag that is counted

Return counter (Int64) - The total amount of times the tag appears in the
    prediction sets.
"""
function tag_freq(tag)

    counter = 0
    for i in sets
        for j in i
            if j == tag
                counter += 1
            end
        end
    end
    return counter
end

"""
    BLSTM(x)

Bidirectional Long Short-Term Memory layer.

Parameter x (Array{Float32, 2}) - X value from the training dataloader.

Return res (Array{Float32, 2}) - Matrix of the forward LSTM predictions
concatenated with backward LSTM predictions.
"""
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

"""
    loss(x, y)

Loss function for model.

Parameter x (Array{Float32, 2}) - X value from the training dataloader.
Parameter y (Array{Float32, 2}) - Y value from the training dataloader.

Return l (Float64) - Loss value.
"""
function loss(x, y)
    Flux.reset!(forward)
    Flux.reset!(backward)
    l = Flux.crossentropy(model(x), y)
    return l
end

"""
    load_parameters!(weights)

Loads trained weights from GPU.

Parameter weights (Vector{Any}) - Vector of updated parameters for each layer
    of the neural network.
"""
function load_parameters!(weights)
    Flux.loadparams!(forward, weights[1])
    Flux.loadparams!(backward, weights[2])
    Flux.loadparams!(embedding, weights[3])
    Flux.loadparams!(predictor, weights[4])
end

function toPval(scores,a_i)
    a_i = sort(a_i)
    L = length(a_i)
    pvaltemp = []
    for x in scores
        push!(pvaltemp,1-((searchsortedfirst(a_i, 1 - x)/length(a_i))))
    end
    return pvaltemp
end

function greatorVec(vec)
    return vec .> epsilon
end

function returnIndex(vec, ind)
    return vec[ind]
end

function notVec(vec)
    return abs.(1 .- vec)
end
