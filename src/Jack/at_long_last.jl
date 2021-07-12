using JLD
using Embeddings
using DataFrames
using Lathe.preprocess: TrainTestSplit
using Flux
using LinearAlgebra
using Plots

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

#---------------------------------------------------------------------#

#------------------Make 3D tensor-------------------------------------#


#----------Beginning of NN training-----------------------------------#
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

acc = 0
for (x, y) in calibrateDL
    acc += norm(y - nn(x))^2
end
mse_acc = acc / length(calibrateDL.data[1][1,:])

function loss(x, y)
    return -1*abs((nn(x)⋅y) / (norm(nn(x))*norm(y)))
end

function loss2(x, y)
    z = norm(nn(x) - y)
    if z < 0
        println(x)
    end
    return z
end

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
