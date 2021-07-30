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
    using Transformers
    using Transformers.Basic
    using Transformers.Pretrain


"""
    wordMasker(Vector{Vector{String}}, Vector{Vector{String}})

takes in a Vector{Vector{String}} words and Vector{Vector{String}} POS
masks a random one with "/MASK/", this returns a Vector{Vector{String}}
aswell as the words which are actually maskes at their POS

Ex.
julia > wordMasker([["a","man","walks","into","a","bar"],
                                        ["a","man","walks","into","a","bar"]])
    ["a","/Mask/","walks","into","a","bar"],
                                        ["a","man","walks","into","/MASK/","bar"]
"""
function wordMasker(sentences, tags)
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
    toBertData(Vector{Vector{String}}, BertModel, wordpeice, tokenizer, Vocabulary, Vector{String})

takes in the output from wordMasker, a vector of the masked words POS,
the BERT Model, wordpeice, tokenizer and vocab from transformers.jl,
and a list of all possible parts of speech to make the BERT embeddings
for the masked word along with a onehot vector for POS
"""
function toBertData(sentences, POS, bert_model, wordpiece, tokenizer, vocab, U)
    sentencesVecs = []
    POSVecs = []
    for i in ProgressBar(1:length(sentences))
        splitind = findall(x -> x == "/MASK/", sentences[i])[1]
        text1 = ""
        text2 = ""
        for ii in 1:length(sentences[i])
            if ii < splitind
            text1 = text1 * " " * sentences[i][ii]
            end
            if ii > splitind
            text2 = text2 * " " * sentences[i][ii]
            end
        end
        text1 = text1 |> tokenizer |> wordpiece
        text2 = text2 |> tokenizer |> wordpiece

        text = ["[CLS]"; text1; "[MASK]"; text2; "[SEP]"]
        ind = findall(x -> x == "[MASK]", text)
        token_indices = vocab(text)
        segment_indices = [fill(1, length(text1)+length(text2) + 3);]

        sample = (tok = token_indices, segment = segment_indices)

        bert_embedding = sample |> bert_model.embed
        feature_tensors = bert_embedding |> bert_model.transformers

        push!(sentencesVecs, feature_tensors[:,ind])

        push!(POSVecs,Flux.onehot(POS[i],U))
    end

    return [sentencesVecs,POSVecs]
end

"""
    toBert(Vector{String}, BertModel, wordpeice, tokenizer, Vocabulary)

getBertEmbedding but for a single sentence and no onehot output
"""
function toBert(sentence,bert_model, wordpiece, tokenizer, vocab)
    splitind = findall(x -> x == "/MASK/", sentence)[1]
    text1 = ""
    text2 = ""
    for ii in 1:length(sentence)
        if ii < splitind
        text1 = text1 * " " * sentence[ii]
        end
        if ii > splitind
        text2 = text2 * " " * sentence[ii]
        end
    end
    text1 = text1 |> tokenizer |> wordpiece
    text2 = text2 |> tokenizer |> wordpiece

    text = ["[CLS]"; text1; "[MASK]"; text2; "[SEP]"]
    ind = findall(x -> x == "[MASK]", text)
    token_indices = vocab(text)
    segment_indices = [fill(1, length(text1)+length(text2) + 3);]

    sample = (tok = token_indices, segment = segment_indices)

    bert_embedding = sample |> bert_model.embed
    feature_tensors = bert_embedding |> bert_model.transformers
    return feature_tensors[:,ind]
end

"""
    vecvec_to_matrix(Vector{Vector})
converts a Vector{Vector} to Matrix
"""
function vecvec_to_matrix(vecvec)
    dim1 = length(vecvec)
    dim2 = length(vecvec[1])
    my_array = zeros(Float32, dim1, dim2)
    for i in ProgressBar(1:dim1)
        for j in 1:dim2
                my_array[i,j] = vecvec[i][j]
        end
    end
    return my_array
end

"""
    minNorm(Vector{Vector{Float32}}, Vector{Float32})

calculates the minimum from a single vector to set of a vector
"""
function minNorm(P, C)
    normz = []
    for p in P
        push!(normz, norm(p - C))
    end
    return minimum(normz)
end

"""
    minNormEx(Vector{Vector{Float32}}, Vector{Float32})

calculates the minimum from a single vector to set of a vector
and insuring that we are not comparing the same vector
"""
function minNormEx(P, C)
    normz = []
    for p in P
        if p != C
            push!(normz, norm(p - C))
        end
    end
    return minimum(normz)
end

"""
    to2GramsFastText(String,Dict(String => Vector{Float32}))

converts a vector of strings not in the embtable by
building it ouf 2-grams or 1-grams, and returns a vector
"""
function to2GramsFastText(word,embtable)
    Grams = []
    for i in 1:length(word)-1
        push!(Grams, word[i:i+1])
    end
    vec = zeros(Float32, 300)
    for x in Grams
        vec += get(embtable, x, zeros(300))
    end
    if vec == zeros(300) || isnan(vec[1])
        Grams = []
        for i in 1:length(word)
            push!(Grams, word[i])
        end
        for x in Grams
            vec += get(embtable, string(x), zeros(300))
        end
        return vec
    end
    return (2 + rand()) .* vec ./ norm(vec)
end

"""
    embedVector(Vector{String},Vector{Vector{Float32}})

converts a vector of strings into a vector of each string's embeddings
"""
function embedVector(vector,embtable)
    embvec = []
    for x in vector
        if x != "/MASK/"
            push!(embvec, embtable[x])
        end
    end
    return embvec
end

"""
    getVocab(Dict(String => Vector{String}),Vector{String})

generates the correct words to use based on POS prediction
"""
function getVocab(POSExamples, PSet)
    newVocab = []

    for x in PSet
        append!(newVocab, get(POSExamples,x,[]))
    end
    return unique(newVocab)
end

function TrainCalibTest(V, p ,q)
    trainingInd = sample(1:size(V)[1],Int(floor(p * size(V)[1])),replace = false)
    extraInd = filter(x->x∉trainingInd, 1:size(V)[1])
    trainV = V[trainingInd]
    extraV = V[extraInd, :]

    calibInd = sample(1:size(extraV)[1],Int(floor((q/(1-p)) * size(extraV)[1])),replace = false)
    testingInd = filter(x->x∉calibInd, 1:size(extraV)[1])
    calibV = V[calibInd, :]
    testV = V[testingInd, :]

    return [trainV, calibV, testV]
end

function SplitVector(V, p)
    trainingInd = sample(1:size(V)[1],Int(floor(p * size(V)[1])),replace = false)
    extraInd = filter(x->x∉trainingInd, 1:size(V)[1])
    trainV = V[trainingInd]
    extraV = V[extraInd]

    return [trainV, extraV]
end


function toBertNoMask(sentence,bert_model, wordpiece, tokenizer, vocab, splitind)
    text1 = ""
    text2 = ""
    for ii in 1:length(sentence)
        if ii < splitind
        text1 = text1 * " " * sentence[ii]
        end
        if ii > splitind
        text2 = text2 * " " * sentence[ii]
        end
    end

    text1 = text1 |> tokenizer |> wordpiece
    text2 = text2 |> tokenizer |> wordpiece


    text = ["[CLS]"; text1; sentence[splitind]; text2; "[SEP]"]

    ind = length(text1) + 2
    token_indices = vocab(text)
    segment_indices = [fill(1, length(text1)+length(text2) + 3);]

    sample = (tok = token_indices, segment = segment_indices)

    bert_embedding = sample |> bert_model.embed
    feature_tensors = bert_embedding |> bert_model.transformers
    return feature_tensors[:,ind]
end
