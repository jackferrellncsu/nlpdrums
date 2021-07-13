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

# Reading in text file
og_text = open("/Users/eplanch/Downloads/1342-0.txt", "r")
corp = read(og_text, String)
close(og_text)

# ------------------------------------------------ #
# -------------- Cleaning the data --------------- #
# ------------------------------------------------ #

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
get_word_index = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab))

# Creation of PridePrej JLD file

JLD.save("PridePrej.jld", "corpus", corp, "sentances", word_per_sent, "data", hcat(sentances, nextword), "embtable", get_word_index)


# ------------------------------------------------ #
# ------------- Save and Load Matrix ------------- #
# ------------------------------------------------ #

# Loads the PridePrej JLD file in
pride_jld = JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/PridePrej.jld")
embtable2 = pride_jld["embtable"]


# Creating matrix for every sentence combination
# Max Pooling
max_mat = zeros(300, length(sentances))
for i in 1:length(sentances)
    mat = zeros(300, length(sentances[i]))
    for j in 1:length(sentances[i])
        mat[:, j] = get(embtable2, sentances[i][j], zeros(300))
    end
    max_vec_indicies = []
    max_vec = []
    for jj in 1:300
        push!(max_vec_indicies, argmax(abs.(mat[jj, :])))
        push!(max_vec, mat[jj, max_vec_indicies[jj]])
    end
    max_mat[:,i] = max_vec'
    println(i)
end

# Creating the "y" values for DataLoader
nextword_emb = zeros(300, length(nextword))
for i in 1:length(nextword)
    nextword_emb[:, i] = get(embtable2, nextword[i], zeros(300))
    println(i)
end

# Saving the max pre-sentence embedding matrices
JLD.save("maxpool_emb.jld", "max_matrix", max_mat, "nextword_matrix", nextword_emb)

# Loading in max matrix
max_mat_jld = JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/maxpool_emb.jld")
maxpool_mat = max_mat_jld["max_matrix"]
nextword_mat = max_mat_jld["nextword_matrix"]
# maxpool_mat over nextword_mat
big_mat = vcat(maxpool_mat, nextword_mat)
maxpool_df = DataFrame(big_mat')

# ------------------------------------------------ #
# --------------- Neural Net prep ---------------- #
# ------------------------------------------------ #

# Test/train split and Train/Calibration split
max_train2, max_test_df = TrainTestSplit(maxpool_df, .8)
max_train_df, max_calib_df = TrainTestSplit(max_train2, .95)

# Converting train/test/calib dataframes to matrices
max_train_raw = Matrix{Float64}(max_train_df)'
max_test_raw = Matrix{Float64}(max_test_df)'
max_calib_raw = Matrix{Float64}(max_calib_df)'

# Dataloader classes
max_train_class = max_train_raw[301:end, :]
max_test_class = max_test_raw[301:end, :]
max_calib_class = max_calib_raw[301:end, :]

# Dataloader matrices
max_train = max_train_raw[1:300, :]
max_test = max_test_raw[1:300, :]
max_calib = max_calib_raw[1:300, :]

# Creation of DataLoader objects
dl_calib = Flux.Data.DataLoader((max_calib, max_calib_class))
dl_test = Flux.Data.DataLoader((max_test, max_test_class))
dl_train = Flux.Data.DataLoader((max_train, max_train_class),
                                    batchsize = 100, shuffle = true)


# ------------------------------------------------ #
# ----------------- Neural Net ------------------- #
# ------------------------------------------------ #

# Neural Net Architecture
nn = Chain(
    Dense(300, 400, relu),
    Dense(400, 500, relu),
    Dense(500, 400, relu),
    Dense(400, 300, x->x)
    )

# Optimizer
opt = RADAM()

# Parameters
ps = Flux.params(nn)

# Loss Function
function loss(x, y)
     return norm(nn(x) - y)
end

# Training the Neural Net, Tracking Loss Progression
totalLoss = []
traceY = []
for i in 1:10
    Flux.train!(loss, ps, dl_train, opt)
    println(i)
    for (x,y) in dl_train
        totalLoss = loss(x,y)
    end
    push!(traceY, totalLoss)
end

# Checking accuracy
acc = 0.0
for (x,y) in dl_calib
    acc += (norm(y - nn(x))).^2
    println(acc)
end
mse_acc = acc / length(max_calib[1,:])

# Trace Plot 1 (loss vs epochs)
x = 1:10
y = traceY
plotly()
plot(x, y, label = "Loss Progression")
xlabel!("Total # of epochs")
ylabel!("Loss values")

# ------------------------------------------------ #
# ------------------ Conformal ------------------- #
# ------------------------------------------------ #
predictions = []
true_values = []
for (x, y) in dl_calib
    push!(predictions, nn(x))
    push!(true_values, y)
end

#=
desiredWord = "the"
ind,vals = toVecs(getValues(S, [get(D, desiredWord, 0), -1]))
vectors = []
size = 0
for i in 1:length(vals)
    push!(vectors, vals[i] * get_vector(M, uni[ind[i]]))
    size += vals[i]
end

avrVector = sum(vectors)./size

a_is = []
for i in 1:length(vals)
    println(i/length(vals))
    if uni[i] in vocabulary(M) && get(S, [get(D, desiredWord, 0), get(D, uni[i], 0)],0 ) != 0
        for ii in 1:toVecs(getValues(S, [get(D, desiredWord, 0), get(D, uni[i], 0)]))[2][1]
            push!(a_is, norm(avrVector - get_vector(M, uni[i])))
        end
    end
end

pval = Vector{Float64}(undef,0)
    words = []
    for i in 1:length(uni)
        println(i / length(uni))
        if uni[i] in vocabulary(M)
            push!(words, uni[i])
            push!(pval,sum(a_is .>= norm(avrVector - get_vector(M, uni[i]))) / (length(a_is) + 1))
        end
    end


epsilon = .95
wordpred = []
    for i in 1:length(pval)
        if pval[i] > epsilon
            push!(wordpred, words[i])
        end
    end
    println("")
    println(length(wordpred) / length(pval))
    println(length(wordpred) / length(pval))
    #println(wordpred)
=#
