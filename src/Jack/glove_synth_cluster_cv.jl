using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--opt1"
            help = "an option with an argument"
        "--opt2", "-o"
            help = "another option with an argument"
            arg_type = Int
            default = 0
        "--flag1"
            help = "an option without argument, i.e. a flag"
            action = :store_true
        "arg1"
            help = "a positional argument"
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

using Glowe
using Lathe.preprocess: TrainTestSplit
using Embeddings
using LinearAlgebra
using Flux
using Plots
using JLD
using Random
using CSV
using DataFrames



function SampleEmbeddings(df, vec_size)
    embed = 0
    embed_mat = Matrix{Float64}(I, vec_size, length(eachrow(df)))
    for (i, r) in enumerate(eachrow(df))
        doc = split(r[1])
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

n = parse(Int64, get(parsed_args, "arg1", 0 ))


param = Int(ceil(n / 250))

seed = n % 100
Random.seed!(seed)
SAMPLE = CSV.read("JonsTraining.csv", DataFrame)



trainTestSplitPercent = .9
batchsize_custom = 100
epochs = 1000


obj = load("embtable.jld")

#load the proper embtable
embtable = obj["embtable" * string(param)]


#prepare embeddings for use
get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
vec_length = length(embtable.embeddings[:, get_word_index["the"]])

#split sample
train, test = TrainTestSplit(SAMPLE, 0.9)

trainEmbs = SampleEmbeddings(train, vec_length)
testEmbs = SampleEmbeddings(test, vec_length)

#Get classifications for train/val/test
classTrain = train[:, 2]
classTest = test[:, 2]

#convert to flux compatible data type
trainDL = Flux.Data.DataLoader((trainEmbs, classTrain'),
                                batchsize = batchsize_custom,
                                shuffle = true)
testDL = Flux.Data.DataLoader((testEmbs, classTest'))

reduc1 = Int(floor(vec_length*0.66))
reduc2 = Int(floor(reduc1*0.66))
reduc3 = Int(floor(reduc2*0.5))
nn = Chain(Dense(vec_length, reduc1, mish),
            Dense(reduc1, reduc2, mish),
            Dense(reduc2, reduc3, mish),
            Dense(reduc3, 10, mish),
            Dense(10, 1, x->Flux.σ.(x)))

opt = Flux.Optimiser(ExpDecay(0.01, 0.9, 200, 1e-4), RADAM())
ps = Flux.params(nn)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))

traceY = []
traceY2 = []
epochs = 500
    for i in 1:epochs
        print(i)
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

save("Errors" * string(param) * "_" * string(seed) * ".jld", "val", errors)
