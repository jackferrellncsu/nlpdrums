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
using Plotly

# Reading in text file
og_text = open("/Users/ckapper/Downloads/1342-0.txt", "r")
corp = read(og_text, String)
close(og_text)

# ------------------------------------------------ #
# -------------- Cleaning the data --------------- #
# ------------------------------------------------ #


#No need to run this again
    # Removing "\r" and "\n"
corp = replace(corp, "\r" => "")
corp =replace(corp, "\n" => "")
# Removing the chapter labels
for i in 1:100
    corp = replace(corp, "chapter "*string(i) => "")
end

# Removing punctuation
corp = replace(corp, "”" => "")
corp = replace(corp, "“" => "")
corp = replace(corp, ";" => "")
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

# Creates a vector with the elements being each sentence in the text
sent_vec = convert(Vector{String},split(corp, "."))

# Creates a vector with each element being a vector
# of individual words in a singular sentence
word_per_sent = []
for i in 1:length(sent_vec)
    sent = convert(Vector{String},split(sent_vec[i], " "))
    push!(word_per_sent, append!(sent, [String(".")]))
    word_per_sent[i] = filter(x->x≠"",word_per_sent[i])
end

# Removes last part of text file that is not actually the book
word_per_sent = word_per_sent[1:end-219]

# Creates a vectors with each entry being an individual word in the corpus
word_count = convert(Vector{String},split(corp, " "))
word_count = filter(x->(x≠"" && x≠"."),word_count)
word_count = word_count[1:findall(x -> x == ".***", word_count)[1]-1]

# Creates a dictionary with each word and its index
uni = convert(Vector{String},unique(word_count))
D = Dict(uni .=> 1:length(uni))

# Creates possible pre-sentence embeddings and
# possible next words
sentances = []
nextword = []
for i in 1:length(word_per_sent)
    println(i/length(word_per_sent))
    for ii in 1:length(word_per_sent[i])-1
        push!(sentances,word_per_sent[i][1:ii])
        push!(nextword,word_per_sent[i][ii+1])
    end
end
# Loading in glove embeddings, creating embedding table
embtable = load_embeddings(GloVe{:en},4, keep_words=Set(uni))
JLD.save("embtable.jld", "embtable", embtable)
t = JLD.load("embtable.jld")
embtable = t["embtable"]
#get vector from word
get_vector_word = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab))
#get word from vector
get_word_vector = Dict(embtable.embeddings[:, ii] => word for (ii, word) in enumerate(embtable.vocab))
get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
# Creation of PridePrej JLD file
JLD.save("PridePrej.jld", "corpus", corp, "sentances", word_per_sent, "data",
        hcat(sentances, nextword), "embtable", embtable, "index_table", embtable_index)


# ------------------------------------------------ #
# ------------- Conformal Playground ------------- #
# ------------------------------------------------ #

# Loads the PridePrej JLD file in
pride_jld = JLD.load("PridePrej.jld")
embtable2 = pride_jld["embtable"]


# Creates embedded vectors for pre-sentences and next words
sentancesVecs = []
nextwordVecs = []
for i in 1:length(sentances)
    push!(sentancesVecs, toEmbedding(sentances[i], embtable_index))
    push!(nextwordVecs, toEmbedding([nextword[i]], embtable_index))
end

sentemb_mat = zeros(Float32, 300, length(sentancesVecs))
wordemb_mat = zeros(Float32, 300, length(sentancesVecs))
for i in 1:length(sentancesVecs)
    sentemb_mat[:, i] = sentancesVecs[i]
    wordemb_mat[:, i] = nextwordVecs[i]
end

# Saving the embedding matrices
JLD.save("emb_matrices.jld", "word_embed_matrix", wordemb_mat, "sent_embed_matrix", sentemb_mat)


# Loading in emebdding matrices
embedding_matrics_jld = JLD.load("emb_matrices.jld")
sentemb_mat = embedding_matrics_jld["sent_embed_matrix"]
wordemb_mat = embedding_matrics_jld["word_embed_matrix"]

#need rows as observations for DataFrame command
sentemb_mat = sentemb_mat'
wordemb_mat = wordemb_mat'

#Need to concatonate sentence and word embeddings so the same rows of each
#are in train/calib/test sets
sent_and_word_emb = hcat(sentemb_mat, wordemb_mat)

df = DataFrame(sent_and_word_emb)

# Test/train split
train_old, test = TrainTestSplit(df, .8)

# Train/Calibration split
train, calib = TrainTestSplit(train_old, .8)

sentemb_train = Matrix{Float64}(train[:, 1:300])
sentemb_calib = Matrix{Float64}(calib[:, 1:300])
sentemb_test = Matrix{Float64}(test[:, 1:300])

wordemb_train = Matrix{Float64}(train[:, 301:600])
wordemb_calib = Matrix{Float64}(calib[:, 301:600])
wordemb_test = Matrix{Float64}(test[:, 301:600])

# Making DataLoader objects to hold data (switch back to columns as obs)
trainDL = Flux.Data.DataLoader((sentemb_train', wordemb_train'),
                            batchsize = 300,
                            shuffle = true)
calibrateDL = Flux.Data.DataLoader((sentemb_calib', wordemb_calib'))

testDL = Flux.Data.DataLoader((sentemb_test', wordemb_test'))


#Make and train neural network
nn = Chain(
    Dense(300, 500, mish),Dense(500, 500, mish),
    Dense(500, 300, x -> x)
    )
opt = RADAM()
ps = Flux.params(nn)

epochs = 100
trace = TrainNN!(epochs)

#Trace Plot (does not work on my laptop)
plotly()
plot(1:epochs, trace)

#Calculate error and MSE
err = 0
for (x, y) in calibrateDL
    err += norm(y - nn(x))^2
end
println(err)
mse = err/length(calibrateDL.data[1])

#Save function for neural net as BSON
using BSON: @save
@save "basic.bson" nn

using BSON: @load
@load "basic.bson" nn

#test nn
test = ConfPred(nn)

#Accuracy/validity of test
accuracy = CheckValidity(test)

# --------------------------- Functions ----------------------------- #
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
    loss(x, y)

Return norm of distance between prediction and actual value
"""
function loss(x, y)
    return norm(nn(x) - y)
    if z < 0
        println(x)
    end
    return z
end

"""
    TrainNN!(epochs)

Train neural network for specified number of epochs
"""
function TrainNN!(epochs)
    trace_vec = []
    for i in 1:epochs
        Flux.train!(loss, ps, trainDL, opt)
        println(i)
        totalLoss = 0
        for (x, y) in trainDL
            totalLoss += loss(x, y)
        end
        push!(trace_vec, totalLoss)
    end
    return trace_vec
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
        print(length(region))
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
#---------------------

#desiredWord = "the"
#ind,vals = toVecs(getValues(S, [get(D, desiredWord, 0), -1]))
#vectors = []
#size = 0
#for i in 1:length(vals)
#    push!(vectors, vals[i] * get_vector(M, uni[ind[i]]))
#    size += vals[i]
#end
#
#avrVector = sum(vectors)./size
#
#a_is = []
#for i in 1:length(vals)
#    println(i/length(vals))
#    if uni[i] in vocabulary(M) && get(S, [get(D, desiredWord, 0), get(D, uni[i], 0)],0 ) != 0
#        for ii in 1:toVecs(getValues(S, [get(D, desiredWord, 0), get(D, uni[i], 0)]))[2][1]
#            push!(a_is, norm(avrVector - get_vector(M, uni[i])))
#        end
#    end
#end
#
#pval = Vector{Float64}(undef,0)
#    words = []
#    for i in 1:length(uni)
#        println(i / length(uni))
#        if uni[i] in vocabulary(M)
#            push!(words, uni[i])
#            push!(pval,sum(a_is .>= norm(avrVector - get_vector(M, uni[i]))) / (length(a_is) + 1))
#        end
#    end
#
#
#epsilon = .95
#wordpred = []
#    for i in 1:length(pval)
#        if pval[i] > epsilon
#            push!(wordpred, words[i])
#        end
#    end
#    println("")
#    println(length(wordpred) / length(pval))
#    println(length(wordpred) / length(pval))
#    #println(wordpred)
#
