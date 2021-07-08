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

# Reading in text file
og_text = open("/Users/eplanch/Downloads/1342-0.txt", "r")
corp = read(og_text, String)
close(og_text)

# ------------------------------------------------ #
# -------------- Cleaning the data --------------- #
# ------------------------------------------------ #

#=
No need to run this again
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
    get_word_index = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab)

    # Creation of PridePrej JLD file
    JLD.save("PridePrej.jld", "corpus", corp, "sentances", word_per_sent, "data", hcat(sentances, nextword), "embtable", embtable), here for reference
=#

# ------------------------------------------------ #
# ------------- Confomral Playground ------------- #
# ------------------------------------------------ #

# Loads the PridePrej JLD file in
pride_jld = JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/PridePrej.jld")
embtable_index = pride_jld["embeddings"]

# Function that takes in words to embed and the index of their embeddings
# and creates embeddings
function toEmbedding(words, Embeddings)
    V = zeros(length(get(Embeddings,"the",0)))
    default = zeros(length(get(Embeddings,"the",0)))
    for x in words
        V += get(Embeddings,x,default)
    end
    return convert(Vector{Float32},V)
end

# Creates embedded vectors for pre-sentences and next words
sentancesVecs = []
nextwordVecs = []
for i in 1:length(sentances)
    push!(sentancesVecs, toEmbedding(sentances[i], embtable_index))
    push!(nextwordVecs, toEmbedding([nextword[i]], embtable_index))
end

# No need to run again, matrix creation

# Create embedding matrix for sentence and next word embeddings
    sentemb_mat = convert(Matrix{Float32}, zeros(300, length(sentancesVecs)))
    wordemb_mat = convert(Matrix{Float32}, zeros(300, length(nextwordVecs)))
    for i in 1:length(sentancesVecs)
        if i == 1
            sentemb_mat = sentancesVecs[i]
            wordemb_mat = nextwordVecs[i]
        else
            sentemb_mat = hcat(sentemb_mat, sentancesVecs[i])
            wordemb_mat = hcat(wordemb_mat, nextwordVecs[i])
        end
        println(string(round(i/length(sentancesVecs), digits=3) * 100)*"%")
    end

    # Saving the embedding matrices
    JLD.save("emb_matrices.jld", "word_embed_matrix", wordemb_mat, "sent_embed_matrix", sentemb_mat)


# Loading in emebdding matrices
embedding_matrics_jld = JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/emb_matrices.jld")
sentemb_mat = embedding_matrics_jld["sent_embed_matrix"]
wordemb_mat = embedding_matrics_jld["word_embed_matrix"]

function split_matrix(mat, train_test, train_calib)
    train_size_old = Int(train_test * length(mat[1,:]))
    test_size = Int(length(mat[1,:]) - train_size)
    train_size = Int(train_calib * train_size_old))
    calib_size = Int(train_size_old - train_size)

    rand_train_array = rand([1:length(mat[1,:]);], train_size)
    for i in 1:length(mat[1,:])
        word_choice = rand_train_array[i]
        mat_col = mat[:, word_choice]

# Test/train split
sentemb_train_old, sentemb_test = TrainTestSplit(sentemb_mat, .8)
wordemb_train_old, wordemb_test = TrainTestSplit(wordemb_mat, .8)
sentemb_train, sentemb_calib = TrainTestSplit(sentemb_train_old, .8)
wordemb_train, wordemb_calib = TrainTestSplit(wordemb_train_old, .8)


# Train/Calibration split

nn = Chain(
    Dense(300, 500, gelu),Dense(500, 500, gelu),
    Dense(500, 500, gelu),Dense(500, 300)
    )
opt = RADAM()
ps = Flux.params(nn)

function loss(x, y)
  if rand() <= .05
     return norm(nn(x) - y)
  else
     return 0
  end
end

for i in 1:1000
    println(i)
    Flux.train!(loss, ps, zip(sentancesVecs, nextwordVecs), opt)
end
#---------------------

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
