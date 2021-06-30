using Glowe
using Lathe
using Embeddings
using LinearAlgebra
using Flux
using Plots
using JLD

include("../data_cleaning.jl")
include("../embeddings_nn.jl")

MED_DATA = importClean()

field = " Cardiovascular / Pulmonary"
sub = filtration(MED_DATA, field)

trainVal, test = TrainTestSplit(sub, 0.9)

#train, validate = TrainTestSplit(trainVal, 0.9)

#Load embeddings - 6B/200D
embtable = load_embeddings(GloVe{:en}, 3, max_vocab_size = 50000)
#6B/300D
embtable2 = load_embeddings(GloVe{:en}, 4, max_vocab_size = 50000)
#42B/300D
embtable3 = load_embeddings(GloVe{:en}, 5, max_vocab_size = 50000)
#840B/300D
embtable4 = load_embeddings(GloVe{:en}, 6, max_vocab_size = 50000)

jldopen("embtable.jld", "w") do file
    write(file, "embtable1", embtable)
    write(file, "embtable2", embtable2)
    write(file, "embtable3", embtable3)
    write(file, "embtable4", embtable4)
end
t = load("embtable.jld")
t["embtable"]
#create dict for embeddings
const get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
const vec_length = length(embtable.embeddings[:, get_word_index["the"]])

#Get embeddings for each set
trainEmbs = SampleEmbeddings(train, vec_length)
testEmbs = SampleEmbeddings(test, vec_length)
#valEmbs = SampleEmbeddings(validate, vec_length)

#Get classifications for train/val/test
classTrain = train[:, 1] .== field
classTrain = classTrain * 1.0

classTest = test[:, 1] .== field
classTest = classTest * 1.0

#classVal = validate[:, 1] .== field
#classVal = classVal * 1.0

batchsize_custom = 100
trainDL = Flux.Data.DataLoader((trainEmbs, classTrain'),
                                batchsize = batchsize_custom,
                                shuffle = true)
#valDL = Flux.Data.DataLoader((valEmbs, classVal'))
testDL = Flux.Data.DataLoader((testEmbs, classTest'))

nn = Chain(Dense(200, 150, mish),
            Dense(150, 100, mish),
            Dense(100, 50, mish),
            Dense(50, 25, mish),
            Dense(25, 10, mish),
            Dense(10, 5, hardσ),
            Dense(5, 1, x->Flux.σ.(x)))

opt = Flux.Optimiser(ExpDecay(0.01, 0.9, 200, 1e-4), RADAM())
ps = Flux.params(nn)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))


totalLoss = []
traceY = []
traceY2 = []
epochs = 500
    for i in 1:epochs
        Flux.train!(loss, ps, trainDL, opt)
        if i % 100 == 0
            println(i)
        end
        for (x,y) in trainDL
            totalLoss = loss(x,y)
        end
        push!(traceY, totalLoss)
        acc = 0
        for (x,y) in testDL
            acc += sum((nn(x) .> .5) .== y)
        end
        decimalError = 1 - acc/length(classVal)
        percentError = decimalError * 100
        percentError = round(percentError, digits=2)
        push!(traceY2, percentError)
    end

#testing for accuracy
acc = 0
    for (x,y) in testDL
        acc += sum((nn(x) .> .5) .== y)
    end
    decimalError = 1 - acc/length(classTest)
    percentError = decimalError * 100
    percentError = round(percentError, digits=2)
    print("Error Rate: ")
    print(percentError)
    println("%")

x = 1:epochs
y = traceY
plot(x, y, label = "Loss Progression")
xlabel!("Total # of epochs")
ylabel!("Loss values")

x2 = 1:epochs
y2 = traceY2
plot(x2, y2)
xlabel!("Total # of epochs")
ylabel!("Error Rate")

#returns sum of embeddings to feed into nn
function SampleEmbeddings(df, vec_size)
    embed = 0
    embed_mat = Matrix{Float64}(I, vec_size, length(eachrow(df)))
    for (i, r) in enumerate(eachrow(df))
        doc = split(r[3])
        embed = getEmbedding(doc[1])
        for d in doc[2:end]
            embed += getEmbedding(d)
        end
        embed_mat[:, i] = embed
    end

    return embed_mat
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
