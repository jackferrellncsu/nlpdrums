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


# Reading in text file
brown_df = CSV.read("brown.csv", DataFrame)

brown_data = brown_df[4]
raw_sentences = split.(brown_data, " ")

raw_tags = []
    raw_words = []
    for sent in raw_sentences
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
unitags = unique.(raw_tags)
uni = []
for i in ProgressBar(1:length(unitags))
    uni = vcat(uni, unitags[i])
    uni = unique(uni)
end


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

function makeDataPOS(sentences, POS, bert_model, wordpiece, tokenizer, vocab, U)
    sentencesVecs = []
    POSVecs = []
    for i in ProgressBar(1:length(sentences))
        splitind = findall( x -> x == "/MASK/", sentences[i])[1]
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

function minNorm(P, C)
    normz = []
    for p in P
        push!(normz, norm(p - C))
    end
    return minimum(normz)
end


Random.seed!(256)
act_word, act_pos, new_sentences = word_masker(raw_words, raw_tags)

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

_bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)

BERTEMB = makeDataPOS(new_sentences,act_pos,bert_model, wordpiece, tokenizer, vocab, uni)

S, P = vecvec_to_matrix(BERTEMB[1]),vecvec_to_matrix(BERTEMB[2])


trainingInd = sample(1:size(S)[1],50000,replace = false)
extraInd = filter(x->x∉trainingInd, 1:size(S)[1])
trainS, trainP = S[trainingInd,:], P[trainingInd,:]
extraS, extraP = S[extraInd, :], P[extraInd,:]

calibInd = sample(1:size(extraS)[1],6340,replace = false)
testingInd = filter(x->x∉calibInd, 1:size(extraS)[1])
calibS, calibP = S[calibInd, :], P[calibInd,:]
testS, testP = S[testingInd, :], P[testingInd,:]

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
    epsilon = .05
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




DL = Flux.Data.DataLoader(((trainS)',(trainP)'), batchsize = 10000, shuffle = true)

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
eta = .001
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
    a_i = zeros(size(calibS)[1])
    for i in ProgressBar(1:size(calibS)[1])
        a_i[i] = (1 - sum(regnet(calibS[i,:]) .* calibP[i,:]))
        if argmax(calibP[i,:]) == argmax(regnet(calibS[i,:]))
            correct += 1
        end
    end

PPP = []
    correct = []
    eff = []
    epsilon = .1
    Q = quantile(a_i,1-epsilon)
    for ii in ProgressBar(1:size(testP)[1])
        Pred = (1 .- regnet(testS[ii, :])) .< Q
        push!(correct, Pred[argmax(testP[ii,:])] == 1)
        push!(eff, sum(Pred))
    end
    print("         ",1-mean(correct), "         ", mean(eff), "         ", median(eff), "         " ,quantile(eff,.75)-quantile(eff,.25))

print(PPP)

for x in tottags
    println(x)
end

using BSON: @save

BSON.@save "BERTPOSMODEL.bson" regnet

#-----------------
vocab = Vocabulary(wordpiece)

markline(sent) = ["[CLS]"; sent; "[SEP]"]

function preprocess(batch)
    sentence = markline.(wordpiece.(tokenizer.(batch[1])))
    mask = getmask(sentence)
    tok = vocab(sentence)
    segment = fill!(similar(tok), 1)

    label = onehotbatch(batch[2], uni)
    return (tok = tok, segment = segment), uni, mask
end

function get_batch(c::Channel, n=1)
    res = Vector(undef, n)
    for (i, x) ∈ enumerate(c)
        res[i] = x
        i >= n && break
    end
    isassigned(res, n) ? batched(res) : nothing
end

clf = (Chain(
        Dropout(0.1),
        Dense(size(_bert_model.classifier.pooler.W, 1), length(uni)),
        logsoftmax
))

bert_model = gpu(
    set_classifier(_bert_model,
                    (
                        pooler = _bert_model.classifier.pooler,
                        clf = clf
                    )
                  )
)

function loss(data, label, mask=nothing)
    e = bert_model.embed(data)
    t = bert_model.transformers(e, mask)

    p = bert_model.classifier.clf(
        bert_model.classifier.pooler(
            t[:,1,:]
        )
    )

    l = Basic.logcrossentropy(label, p)
    return l, p
end

function train!()
    global Batch
    global Epoch
    @info "start training:"
    for e = 1:Epoch
        @info "epoch: $e"
        datas = datas_tr # Training data generated

        i = 1
        al::Float64 = 0.
        while (batch = get_batch(datas, Batch)) !== nothing
            data, label, mask = todevice(preprocess(batch))
            l, p = loss(data, label, mask)
            # @show l
            a = acc(p, label)
            al += a
            grad = gradient(()->l, ps)
            i+=1
            update!(opt, ps, grad)
            i%16==0 && @show al/i
        end

        test()
    end
end
