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

# ------------------------ Data Cleaning ------------------------ #

#Masks a random word in each sentence
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

#Turning the tags into a bert sentence with [MASK] as the masked word
#We then use the default transformer to create the intented embedding for the word

function makeDataPOS(sentences, POS, bert_model, wordpiece, tokenizer, vocab, U)
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

#The Best Non-Conformity we have
function minNorm(P, C)
    normz = []
    for p in P
        push!(normz, norm(p - C))
    end
    return minimum(normz)
end

function minNormEx(P, C)
    normz = []
    for p in P
        if p != C
            push!(normz, norm(p - C))
        end
    end
    return minimum(normz)
end

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



# Reading in text file
brown_df = CSV.read("brown.csv", DataFrame)

brown_data = brown_df[4]
raw_sentences = split.(brown_data, " ")

#Cleans the tags
raw_tags = []
    raw_words = []
    for sent in ProgressBar(raw_sentences)
    raw_tags_temp = []
    raw_words_temp = []
    for word in sent
        ind = findlast(x -> x == '/', word)
        POS = word[ind+1:end]
        POS = replace(POS, "bez" => "bbb")
        POS = replace(POS, "-hl" => "")
        POS = replace(POS, "-tl" => "")
        POS = replace(POS, "-nc" => "")
        POS = replace(POS, "fw-" => "")
        push!(raw_tags_temp, convert(String,POS))
        push!(raw_words_temp, lowercase(word[1:ind-1]))
    end
    push!(raw_tags, raw_tags_temp)
    push!(raw_words, raw_words_temp)
end

#creates the unique POS tags
unitags = unique.(raw_tags)
uni = []
for i in ProgressBar(1:length(unitags))
    uni = vcat(uni, unitags[i])
    uni = unique(uni)
end


Random.seed!(256)
act_word, act_pos, new_sentences = word_masker(raw_words, raw_tags)

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

#creating the bertmodel
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)

BERTEMB = makeDataPOS(new_sentences,act_pos,bert_model, wordpiece, tokenizer, vocab, uni)

S, P = vecvec_to_matrix(BERTEMB[1]),vecvec_to_matrix(BERTEMB[2])

#Splitting the masked word embedding into training, testing, calibration at ≈.9, .09,.01

trainingInd = sample(1:size(S)[1],50000,replace = false)
extraInd = filter(x->x∉trainingInd, 1:size(S)[1])
trainS, trainP = S[trainingInd,:], P[trainingInd,:]
extraS, extraP = S[extraInd, :], P[extraInd,:]

calibInd = sample(1:size(extraS)[1],6340,replace = false)
testingInd = filter(x->x∉calibInd, 1:size(extraS)[1])
calibS, calibP = S[calibInd, :], P[calibInd,:]
testS, testP = S[testingInd, :], P[testingInd,:]


#---------------------
#Jon's Method
POSSENTVECS = Dict()
    for i in ProgressBar(1:size(trainS)[1])
        part = Flux.onecold(trainP[i,:], uni)
        POSSENTVECS[part] = append!(get(POSSENTVECS, part, []), [trainS[i,:]])
end

POSOccurance = Dict()
    for x in uni
        POSOccurance[x] = length(get(POSSENTVECS,x,[]))
    end

a_i = []
    for i in ProgressBar(1:size(calibS)[1])
        part = Flux.onecold(calibP[i,:], uni)
        P = get(POSSENTVECS, part, 0)
        if P != 0
            X = POSOccurance[part]
            T = log(X)
            push!(a_i, minNorm(P,calibS[i,:])/T)
        end
    end


correct = []
    eff = []
    epsilon = .2
    Q = quantile(a_i, 1-epsilon)
    for i in ProgressBar(1:size(testS)[1])
        pred = []
        for ii in 1:length(uni)
            P = get(POSSENTVECS, uni[ii], 0)
            if  P != 0
                #if norm(mean(P) - testing[i,1:300]) < Q
                X = POSOccurance[uni[ii]]
                T = log(X)
                if minNorm(P,testS[i,:])/T <= Q
                    push!(pred, uni[ii])
                end
            end
        end
        trueWord = Flux.onecold(testP[i,:], uni)
        push!(correct,trueWord in pred)
        push!(eff, length(pred))
        PPP = pred
        print("         ",1-mean(correct), "         ", mean(eff), "         ", median(eff), "         " ,quantile(eff,.75)-quantile(eff,.25))
    end

function minNorm(P, C)
    normz = []
    for p in P
        push!(normz, norm(p - C))
    end
    return minimum(normz)
end
#------------------------
#Bert's Method

ind = sample(1:50000,50000,replace = false)
DL = Flux.Data.DataLoader(((trainS[ind,:])',(trainP[ind,:])'), batchsize = 10000, shuffle = true)

regnet =Chain(
    Dense(768, 500, gelu),
    Dense(500, 400, gelu),
    Dense(400, 300, gelu),
    Dense(300, 190,  x -> x), softmax)

ps = Flux.params(regnet)


function loss(x, y)
    return sum(Flux.Losses.crossentropy(regnet(x),y))
end

testmode!(regnet, false)

losses = []
batch  = 1000
epochs = 1000
eta = .005
opt = RADAM(eta)
for i in ProgressBar(1:epochs)
    Flux.train!(loss, ps, DL, opt)
    L = sum(loss.(eachrow(trainS[1:1000,:]), eachrow(trainP[1:1000,:])))
    #if i > 1 && L > losses[end]
        #eta = eta * .9
        #opt = RADAM(eta)
    #end
    push!(losses,L)
    print("         ", L, "        ", sum(regnet(trainS[1,:]) .* trainP[1,:]),"       ", eta, "      ")
end

testmode!(regnet)

correct = 0
    a_i1 = zeros(size(calibS)[1])
    for i in ProgressBar(1:size(calibS)[1])
        a_i1[i] = (1 - sum(regnet(calibS[i,:]) .* calibP[i,:]))
        if argmax(calibP[i,:]) == argmax(regnet(calibS[i,:]))
            correct += 1
        end
    end


correct = []
    eff = []
    epsilon = .01
    Q1 = quantile(a_i1,1-epsilon)
    for ii in ProgressBar(1:size(testP)[1])
        Pred = (1 .- regnet(testS[ii, :])) .<= Q
        push!(correct, Pred[argmax(testP[ii,:])] == 1)
        push!(eff, sum(Pred))
    end
    println("         ",1-mean(correct), "         ", mean(eff), "         ", median(eff), "         " ,quantile(eff,.75)-quantile(eff,.25))


using BSON: @save

BSON.@save "BERTPOSMODELFinal.bson" regnet

using BSON: @load

BSON.@load "BERTPOSMODELFinal.bson" regnet

#-----------------
brown_files = brown_df[1]
brown_paras = brown_df[2]
brown_files_uni = unique(brown_files)
mapFileToNum = Dict()
for i in 1:length(brown_files_uni)
    mapFileToNum[brown_files_uni[i]] = i
end
brown_in_files = [[] for i=1:length(brown_files_uni)]
for i in ProgressBar(1:length(brown_files))
    append!(brown_in_files[mapFileToNum[brown_files[i]]], raw_words[i])
end

unique_words = unique(Base.Iterators.flatten(brown_in_files))


POSExamples = Dict()
    for i in ProgressBar(1:length(raw_words))
    for ii in 1:length(raw_words[i])
        POSExamples[raw_tags[i][ii]] = append!(get(POSExamples, raw_tags[i][ii], []),[raw_words[i][ii]])
    end
end

unilength = []
for x in uni
    push!(unilength,length(unique(POSExamples[x])))
end

embeddings_fasttext = load_embeddings(FastText_Text{:en}, 2)
embtable = Dict(word=>embeddings_fasttext.embeddings[:,ii] for (ii,word) in ProgressBar(enumerate(embeddings_fasttext.vocab)))

missing_words = []
for x in unique_words
    if get(embtable, x, 0) == 0
        push!(missing_words, x)
    end
end

bad = 0
    extra_missing_words = []
    for x in missing_words
    embtable[x] = to2GramsFastText(x, embtable)
    if embtable[x] == zeros(300)
        bad += 1
        push!(extra_missing_words, x)
    end
end
embtable["``"] = embtable["''"]


new_sentences_in_files = [[] for i=1:length(brown_files_uni)]
    for i in ProgressBar(1:length(new_sentences))
        append!(new_sentences_in_files[mapFileToNum[brown_files[i]]], new_sentences[i])
    end


uni_new_sentences_in_files = unique.(new_sentences_in_files)
uni_new_sentences_in_files = filter.(x -> x != "/MASK/", uni_new_sentences_in_files)
uni_new_sentences_embeddings = [[] for i=1:length(brown_in_files)]
for (i,x) in enumerate(uni_new_sentences_in_files)
    for y in x
        push!(uni_new_sentences_embeddings[i], embtable[y])
    end
end

unique_words_embeddings = []
    for (x,i) in zip(unique_words, ProgressBar(1:length(unique_words)))
    push!(unique_words_embeddings,minNorm(uni_new_sentences_embeddings[i], embtable["dislikes"]))
end

maskedindicies = []
for i in 1:500
    push!(maskedindicies, findall(x -> x == "/MASK/", new_sentences_in_files[i]))
end

wordIndTrain = sample(1:500, 420, replace = false)
wordIndExtra = filter(x -> x ∉ wordIndTrain, 1:500)
wordIndCalib = sample(wordIndExtra, 50, replace = false)
wordIndTest = filter(x -> x ∉ wordIndCalib, wordIndExtra)

trainMaskedInd = maskedindicies[wordIndTrain]
calibMaskedInd = maskedindicies[wordIndCalib]
testMaskedInd = maskedindicies[wordIndTest]

a_i2 = []
    for x in ProgressBar(wordIndTrain)
    for y in maskedindicies[x]
        push!(a_i2,minNormEx(uni_new_sentences_embeddings[x], embtable[brown_in_files[x][y]]))
    end
end

epsilon = .1
    sentenceNum = 0
    eff = []
    correct = []
    Q2 = quantile(a_i2, 1-epsilon)
    for x in ProgressBar(wordIndExtra)
    Pred = []
    for y in maskedindicies[x]
        if maskedindicies[x][1] == y
            for z in unique_words
                if minNormEx(uni_new_sentences_embeddings[x], embtable[z]) <= Q2
                    push!(Pred, z)
                end
            end
            push!(eff, length(Pred))
        end
        push!(correct, brown_in_files[x][y] in Pred)
    end
    print( 1-mean(correct), "        ", median(eff), "           ")
    end


e1 = .01
e2 = .09
Q1 = quantile(a_i1, 1-e1)
Q2 = quantile(a_i2, 1-e2)
PSet = []
    V = vec(regnet(toBert(["the","dog", "/MASK/", "to", "fast", ",", "he", "barked", "the", "whole", "time"], bert_model, wordpiece, tokenizer, vocab)))
    Pred = (1 .- V) .<= Q1
    Pset = []
    for i in 1:length(Pred)
    if Pred[i] == 1
        push!(PSet, uni[i])
    end
end

WPSet = []
    MS = Matrix(undef,length(getVocab(POSExamples, PSet)), 2)
    EV = embedVector(["the","dog", "to", "fast", ",", "he", "barked", "the", "whole", "time"],embtable)
    for (i,z) in enumerate(getVocab(POSExamples, PSet))
        M = minNorm(EV, embtable[z])
        MS[i,:] = [M,z]
    if M <= Q2
        push!(WPSet, z)
    end
end

function predictBlank(a_i1, a_i2, e1, e2, transcript, sentence, vocabulary)
    Q1 =
end

function embedVector(vector,embtable)
    embvec = []
    for x in vector
        push!(embvec, embtable[x])
    end
    return embvec
end

function getVocab(POSExamples, PSet)
    newVocab = []
    for x in PSet
        append!(newVocab, POSExamples[x])
    end
    return unique(newVocab)
end
