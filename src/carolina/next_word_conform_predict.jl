using JLD
using Embeddings
using DataFrames
using Lathe.preprocess: TrainTestSplit
using Flux
using LinearAlgebra
using Plots

include("../emi/new_conf.jl")

og_text = open("/Users/ckapper/Downloads/1342-0.txt", "r")
corpus = read(og_text, String)
close(og_text)

# Cleaning data
corp = data_cleaner(corpus)

# Creating vector of sentences
sent_vec = convert(Vector{String},split(corp, "."))

split_sent = split_sentences(sent_vec)

# Creates a vectors with each entry being an individual word in the corpus
word_count = convert(Vector{String},split(corp, " "))
word_count = filter(x->(x≠"" && x≠"."),word_count)

# Creates a dictionary with each word and its index
uni = convert(Vector{String},unique(word_count))
D = Dict(uni .=> 1:length(uni))

# Loading in glove embeddings
embtable = load_embeddings(GloVe{:en},4, keep_words=Set(uni))
#get vector from word
get_vector_word = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab))
#get word from vector
get_word_vector = Dict(embtable.embeddings[:, ii] => word for (ii, word) in enumerate(embtable.vocab))

# Creates possible pre-sentence embeddings and
# possible next words
sentences = []
nextword = []
for i in 1:length(split_sent)
    for ii in 1:length(split_sent[i])-1
        push!(sentences,split_sent[i][1:ii])
        push!(nextword,split_sent[i][ii+1])
    end
end

# Creation of PridePrej JLD file
JLD.save("PridePrej.jld", "corpus", corp, "sentences", split_sent, "data",
        hcat(sentences, nextword), "embtable", embtable)


# Loads the PridePrej JLD file in
pride_jld = JLD.load("PridePrej.jld")
embedding_table = pride_jld["embtable"]
pre_sentence = pride_jld["data"][:, 1]
next_word = pride_jld["data"][:, 2]

# Creates embedded vectors for pre-sentences and next words
sentencesVecs = []
nextwordVecs = []
for i in 1:length(sentences)
    push!(sentencesVecs, toEmbedding(pre_sentence[i], embtable_index))
    push!(nextwordVecs, toEmbedding([next_word[i]], embtable_index))
end

sentemb_mat = zeros(Float32, 300, length(pre_sentence))
wordemb_mat = zeros(Float32, 300, length(pre_sentence))
for i in 1:length(sentencesVecs)
    sentemb_mat[:, i] = sentencesVecs[i]
    wordemb_mat[:, i] = nextwordVecs[i]
end

#need rows as observations for DataFrame command
sentemb_mat = sentemb_mat'
wordemb_mat = wordemb_mat'

#Need to concatonate sentence and word embeddings so the same rows of each
#are in train/calib/test sets
sent_and_word_emb = hcat(sentemb_mat, wordemb_mat)

data_embs = DataFrame(sent_and_word_emb)

#------------------------NN training-----------------------------------#
train, test = TrainTestSplit(data_embs)

proper_train, calibrate = TrainTestSplit(train, 0.95)


train_x_mat = zeros(length(proper_train[1,1]), size(proper_train)[1])
train_y_mat = zeros(length(proper_train[2, 1]), size(proper_train)[1])
calibrate_x_mat = zeros(length(calibrate[1, 1]), size(calibrate)[1])
calibrate_y_mat = zeros(length(calibrate[2, 1]), size(calibrate)[1])

for (i, r) in enumerate(eachrow(proper_train))
    train_x_mat[:, i] = r[1]
    train_y_mat[:, i] = r[2]
end

for (i, r) in enumerate(eachrow(calibrate))
    calibrate_x_mat[:, i] = r[1]
    calibrate_y_mat[:, i] = r[2]
end


trainDL = Flux.Data.DataLoader((train_x_mat, train_y_mat),
                            batchsize = 1000,
                            shuffle = true)
calibrateDL = Flux.Data.DataLoader((calibrate_x_mat, calibrate_y_mat))

nn = Chain(Dense(300, 400, relu),
           Dense(400, 300, x->x))

opt = RADAM(1e-4)
ps = Flux.params(nn)

epochs = 10
trace = TrainNN!(epochs)

plot(1:epochs, trace)


#calculate mean squared error (MSE)
err = 0
for (x, y) in calibrateDL
    err += norm(y - nn(x))^2
end
mse = err / length(calibrateDL.data[1][1,:])

#test nn
test = ConfPred(nn, 0.25)

#Accuracy/validity of test
accuracy = CheckValidity(test)


#---------------------Functions----------------------#

"""
    toEmbedding(words, Embeddings)

Input words to impend & index of their embeddings
Create embeddings
"""
function toEmbedding(words, Embeddings)
    V = zeros(length(get(Embeddings,"the",0)))
    default = zeros(length(get(Embeddings,"the",0)))
    for x in words
        V += get(Embeddings,x,default)
    end
    return convert(Vector{Float32},V)
end

"""
    loss(x,y)

Return -1 * absolute value of (nn(x).y) / norm of nn(x) * norm of y
"""
function loss(x, y)
    return -1*abs((nn(x)⋅y) / (norm(nn(x))*norm(y)))
end

"""
    loss2(x, y)

Return norm of distance between prediction and actual value
"""
function loss2(x, y)
    z = norm(nn(x) - y)
    if z < 0
        println(x)
    end
    return z
end

"""
    SummedContextEmbeddings(mat, vec_length)

"""
function SummedContextEmbeddings(mat, vec_length)
    summed_embs = []
    res_embs = []
    for (i, r) in enumerate(eachrow(mat))
        con_emb = zeros(vec_length)
        res_emb = getEmbedding(r[2])
        for word in r[1]
            con_emb += getEmbedding(word)
        end
        push!(summed_embs, con_emb)
        push!(res_embs, res_emb)
    end
    println(length(summed_embs))
    println(length(res_embs))
    z = hcat(summed_embs, res_embs)
    return DataFrames.DataFrame(z)
end

"""
    getEmbedding(word)

Access an embedding corresponding to a word
"""
function getEmbedding(word)
    if word ∈ keys(get_word_index)
        ind = get_word_index[word]
        emb = embtable.embeddings[:, ind]
    else
        vec_length = length(embtable.embeddings[:, get_word_index["the"]])
        emb = zeros(vec_length)
    end
    return emb
end

"""
    TrainNN!(epochs)

Train neural network for specified number of epochs
"""
function TrainNN!(epochs)
    traceY = []
    for i in 1:epochs
        Flux.train!(loss2, ps, trainDL, opt)
        println(i)
        totalLoss = 0
        for (x,y) in trainDL
         totalLoss += loss2(x,y)
         #println("Total Loss: ", totalLoss)
        end
        push!(traceY, totalLoss)
    end
    return traceY
end

"""
    ConfPred(nn, ϵ = 0.05)

Given a neural net, performes inductive conformal prediction and returns prediction
regions for test set.
"""
function ConfPred(nn, ϵ = 0.05)
    α = Vector{Float64}()
    for (x, y) in calibrateDL
        α_i = norm(y - nn(x))
        push!(α, α_i)
    end
    println("Noncomformity scores calculated")
    all_regs = Vector{Vector{String}}()
    for (x, y) in testDL
        a_k = norm(y - nn(x))
        push!(α, a_k)
        q = quantile(α, 1-ϵ)
        region = Vector{String}()
        pred = nn(x)
        for i in get_vector_word
            dist = norm(pred - i[2])
            if dist <= q
                push!(region, i[1])
            end
        end
        pop!(α)
        push!(all_regs, region)
        println(length(region))
    end
    return all_regs
end

"""
    CheckValidity(intervals)

Checks how many values actually lie in the confidence regions generated by ICP
"""
function CheckValidity(intervals)
    acc = 0
    for (i, region) in enumerate(intervals)
        if mean(wordemb_test'[:,i]) != 0 && get_word_vector[wordemb_test'[:, i]] ∈ region
            acc += 1
        end
    end
    return acc / length(intervals)

end
