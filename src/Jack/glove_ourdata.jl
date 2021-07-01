using Glowe
using Lathe
using Embeddings
using LinearAlgebra
using Flux
using Plots
using JLD
using Random

Random.seed!(13)
include("../data_cleaning.jl")
include("../embeddings_nn.jl")

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

trainTestSplitPercent = .7
batchsize_custom = 100
epochs = 500

errorrates = []
predictions = []
trueValues = []



embtable = load("embtable.jld", "embtable3")

n = parse(Int64, get(parsed_args, "arg1", 0))

MED_DATA = importClean()

field = " Cardiovascular / Pulmonary"
sub = filtration(MED_DATA, field)

Random.seed!(n)
train, test = TrainTestSplit(sub, 0.9)

#create dict for embeddings
const get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
const vec_length = length(embtable.embeddings[:, get_word_index["the"]])

trainEmbs = SampleEmbeddings(train, vec_length)
testEmbs = SampleEmbeddings(test, vec_length)

#Get classifications for train/val/test
classTrain = train[:, 1] .== field
classTrain = classTrain * 1.0

classTest = test[:, 1] .== field
classTest = classTest * 1.0

batchsize_custom = 100

trainDL = Flux.Data.DataLoader((trainEmbs, classTrain'),
                                batchsize = batchsize_custom,
                                shuffle = true)
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


#totalLoss = 0
traceY = []
traceY2 = []
epochs = 1000
    for i in 1:epochs
        Flux.train!(loss, ps, trainDL, opt)
        if i % 100 == 0
            println(i)
        end
        # for (x,y) in trainDL
        #     totalLoss = loss(x,y)
        # end
        # push!(traceY, totalLoss)
        # acc = 0
        # for (x,y) in testDL
        #     acc += sum((nn(x) .> .5) .== y)
        # end
        # decimalError = 1 - acc/length(classVal)
        # percentError = decimalError * 100
        # percentError = round(percentError, digits=2)
        # push!(traceY2, percentError)
    end

acc = 0
    for (x,y) in testDL
        global acc += sum((nn(x) .> .5) .== y)
    end

preds = (nn(testEmbs))
trues = classTest
errors = 1 - acc/length(classTest)

save("Preds" * string(n) * ".jld", "val", preds)
save("Trues" * string(n) * ".jld", "val", classTest)
save("Errors" *string(n) * ".jld", "val", errors)
