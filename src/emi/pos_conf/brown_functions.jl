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

function get_word_vec(sentences)

    words = Vector{String}()
    for i in 1:length(sentences)
        for j in 1:length(sentences[i])
            push!(words, sentences[i][j])
        end
    end
    return words
end

function get_keys(dict)

    keyz = []
    for i in keys(dict)
        push!(keyz, i)
    end
    return keyz
end

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
        # For pre-sentences with < "num_words", average of word embeddings
        # is taken and added to tensor
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

function create_embeddings(masked_word, masked_pos, new_sentences, dict)

    # Finds indices of masked words for each sentence
    masked_ind = []
    for i in 1:length(new_sentences)
        for j in 1:length(new_sentences[i])
            if new_sentences[i][j] == "/MASK/"
                push!(masked_ind, j)
            end
        end
    end

    # Embeddings of masked words, not needed
    masked_embeddings = Vector{Vector{Float32}}()
    for i in 1:length(masked_word)
        push!(masked_embeddings, get(dict, masked_word[i], zeros(300)))
    end

    # Embeddings for each sentence
    sentence_emb = []
    for i in 1:length(new_sentences)
        temp = []
        for j in 1:length(new_sentences[i])
            push!(temp, get(dict, new_sentences[i][j], zeros(300)))
        end
        push!(sentence_emb, temp)
    end


    return masked_ind, masked_embeddings, sentence_emb
end

function splitter(sent_emb, onehot_vecs, train_test, train_calib)

    # Computing sizes of each set
    first_train_size = Int(ceil(Base.length(sent_emb) .* train_test))
    test_size = Int(Base.length(sent_emb) - first_train_size)
    train_size = Int(ceil(first_train_size * train_calib))
    calib_size = Int(first_train_size - train_size)

    train = []
    train_class = []
    train_ind = sample(1:Base.length(sent_emb), train_size, replace = false)
    for i in 1:length(train_ind)
        push!(train, sent_emb[train_ind[i]])
        push!(train_class, onehot_vecs[train_ind[i]])
    end
    train = convert(Vector{Vector{Vector{Float32}}}, train)
    train_class = convert(Vector{Vector{Float32}}, train_class)

    test = []
    test_class = []
    test_ind = sample(1:Base.length(sent_emb), test_size, replace = false)
    for i in 1:length(test_ind)
        push!(test, sent_emb[test_ind[i]])
        push!(test_class, onehot_vecs[test_ind[i]])
    end
    test = convert(Vector{Vector{Vector{Float32}}}, test)
    test_class = convert(Vector{Vector{Float32}}, test_class)


    calib = []
    calib_class = []
    calib_ind = sample(1:Base.length(sent_emb), calib_size, replace = false)
    for i in 1:length(calib_ind)
        push!(calib, sent_emb[test_ind[i]])
        push!(calib_class, onehot_vecs[calib_ind[i]])
    end
    calib = convert(Vector{Vector{Vector{Float32}}}, calib)
    calib_class = convert(Vector{Vector{Float32}}, calib_class)

    return train, train_class, test, test_class, calib, calib_class
end
