using JLD
using Embeddings
using DataFrames
using Lathe.preprocess: TrainTestSplit
using Flux
using LinearAlgebra
using Plotly

obj = load("PridePrej.jld")
    data = obj["data"]
    sentences = obj["sentances"]
    corpus = obj["corpus"]


# ---------- Embeddings ---------------------------#
embtable = load("embtable.jld", "embtable")
get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
vec_length = length(embtable.embeddings[:, get_word_index["the"]])
data_embs = SummedContextEmbeddings(data, vec_length)

# ---------- NN training ---------------------------#
train, test = TrainTestSplit(data_embs, 0.8)

proper_train, calibrate = TrainTestSplit(train, 0.9)

trainDL = Flux.Data.DataLoader((proper_train[1], proper_train[2]),
                            batchsize = 100,
                            shuffle = true)
calibrateDL = Flux.Data.DataLoader((calibrate[1], calibrate[2]))

testDL = Flux.Data.DataLoader((test[1], test[2]))

nn = Chain(Dense(300, 400, mish),
           Dense(400, 300, x -> x))

opt = RADAM(1e-4)
ps = Flux.params(nn)

epochs = 10
trace = TrainNN!(epochs)

Plotly.plot(1:epochs, trace)

err = 0
for (x, y) in calibrateDL
    err += norm(y - nn(x))^2
end
mse = err / length(calibrateDL.data[2,:])



# -------------- Functions (run first) ------------------ #

"""
    SummedContextEmbeddings(mat, vec_length)

Sum pre-sentence embedding vectors before desired next word
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

Retrieve embedding for specific word
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

Train neural network on training data for specified number of epochs
    and print epoch number as they are completed
"""
function TrainNN!(epochs)
    trace = []
    for i in 1:epochs
        Flux.train!(loss, ps, trainDL, opt)
        println(i)
        for (x, y) in trainDL
            totalLoss += loss(x, y)
        end
        push!(totalLoss, trace)
    end
end

function loss(x, y)
    z =  norm(nn(x) - y)
    if z < 0
        println(x)
    end
    return z
end
