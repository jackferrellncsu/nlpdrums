using JLD
using Embeddings
using DataFrames
using Lathe.preprocess: TrainTestSplit
using Flux
using LinearAlgebra

obj = load("PridePrej.jld")
    data = obj["data"]
    sentences = obj["sentances"]
    corpus = obj["corpus"]

# splitted = split(corpus)
# uniques = unique(z)
# filter!(word->!occursin(".", word), uniques)
# embtable = load_embeddings(GloVe{:en}, 6, keep_words = uniques)
# save("pridePrejEmbs.jld", "embtable", embtable)

#----------Preparation of Embeddings----------------------------------#
embtable = load("pridePrejEmbs.jld", "embtable")
get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
vec_length = length(embtable.embeddings[:, get_word_index["the"]])
data_embs = SummedContextEmbeddings(data, vec_length)

#----------Beginning of NN training-----------------------------------#
train, test = TrainTestSplit(data_embs)

proper_train, calibrate = TrainTestSplit(train, 0.95)

trainDL = Flux.Data.DataLoader((proper_train[1], proper_train[2]),
                            batchsize = 100,
                            shuffle = true)
calibrateDL = Flux.Data.DataLoader((calibrate[1], calibrate[2]))

nn = Chain(Dense(300, 400, mish),
           Dense(400, 300, mish))

opt = RADAM(1e-4)
ps = Flux.params(nn)

function loss(x, y)
    return norm(nn(x) - y)
end

TrainNN!(50)






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

function TrainNN!(epochs)
    for i in 1:epochs
        Flux.train!(loss, ps, zip(proper_train[:,1], proper_train[:, 2]), opt)
    end
end
