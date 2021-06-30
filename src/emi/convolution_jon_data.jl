# Reading command line for job index
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

using Flux
using LinearAlgebra
using Statistics
using ROCAnalysis
using MLBase
using Plots
using RecursiveArrayTools
using TextAnalysis
using InvertedIndices
using JLD
using StatsBase
using Random
using CSV
using Lathe.preprocess: TrainTestSplit


# ====================================================== #
# ================= DTM_convolution.jl ================= #
# ====================================================== #
function CreateDTM(true_data)
    rightDocs = []
    wrongDocs = []
    allDocs = []
    class = []

    for i in 1:length(true_data[:, 1])
        #First for loop collects items in field of interest
        if true_data[i, 2] == 1.0
            #Append to document collection
            push!(rightDocs, StripUnwanted((true_data[i, :])))
            push!(allDocs, StripUnwanted((true_data[i, :])))
            push!(class, (true_data[i, 2]))
        else
            #= Collects indeces not included in field of interest=#
            push!(wrongDocs, StripUnwanted((true_data[i, :])))
            push!(allDocs, StripUnwanted((true_data[i, :])))
            push!(class, (true_data[i, 2]))
        end
    end
    #Create corpus with cleaned string documents
    crps = Corpus(rightDocs)
    totalcrps = Corpus(allDocs)

    update_lexicon!(totalcrps)
    lex = lexicon(totalcrps)

    m = DtmHelper(totalcrps, lex)
    m = Vec2Mat(m)
    println("DTM has been created")
    return m
end

function DtmHelper(crps, lex)
    matrix = []
    t = [1]
    f = [0]
    for i in crps
        a = dtv(i, lex)
        a = vec(a)
        if author(i) == "1"
            push!(a, 1)
        else
            push!(a, 0)
        end

        push!(matrix, a)
    end

    return matrix
end

function Vec2Mat(v)
    VA = VectorOfArray(v)
    return(convert(Array, VA))
end

function StripUnwanted(row)
    sd = TextAnalysis.StringDocument(row[1])
    author!(sd, string(row[2]))
    return sd
end

# ====================================================== #
# ================ Beginning of Conv NN ================ #
# ====================================================== #
trainTestSplitPercent = .9
batchsize_custom = 100
epochs = 1000

errorrates = []
predictions = []
trueValues = []

# ====================================================== #
# ==================== Conv For Loop =================== #
# ====================================================== #
j = parse(Int64, get(parsed_args, "arg1", 0 ))

println("Iteration number: "*string(j))
bVal = 1 + ((j * 1000) - 1000)
eVal = 1000 * j
# Imports clean data
true_data = CSV.read("JonsData.csv", DataFrame)
true_data = true_data[bVal:eVal, :]

# Create DTM
DTM = CreateDTM(true_data)
total_DTM = DataFrame(DTM')


Random.seed!(j)
train, test = TrainTestSplit(total_DTM, trainTestSplitPercent)


# Finding classifcation vectors
class_train = train[:, end]
class_test = test[:, end]

# Removing classification columns
dtm_train = Matrix(train[:, 1:end-1])
dtm_test = Matrix(test[:, 1:end-1])

# Convolutional Layer
layers = Chain(
        # Manually reduces 10 (11 - 1) words off of the vocab
        Conv(tuple(1, 12), 1 => 1, relu),
        MaxPool(tuple(1, 15)))

# Convolution & Pooling for training matrix
train1 = length(dtm_train[:,1])
train2 = length(dtm_train[1,:])
train_array = reshape(dtm_train, (train1, train2, 1, 1))
conv_train_array = layers(train_array)
conv_train_mat = conv_train_array[1, :]'
for i in 2:length(conv_train_array[:,1])
    global conv_train_mat = vcat(conv_train_mat, conv_train_array[i,:]')
end

# Convolution & Pooling for testing matrix
test1 = length(dtm_test[:,1])
test2 = length(dtm_test[1,:])
test_array = reshape(dtm_test, (test1, test2, 1, 1))
conv_test_array = layers(test_array)
conv_test_mat = conv_test_array[1, :]'
for i in 2:length(conv_test_array[:,1])
    global conv_test_mat = vcat(conv_test_mat, conv_test_array[i,:]')
end

# Making layers for neural net
L1 = length(conv_test_mat[1,:]) # ~582
L2 = Int(ceil(L1/3) * 2) # ~388
L3 = Int(ceil(L2/2)) # ~194
L4 = Int(ceil(L3/2)) # ~97
L5 = Int(ceil(L4/2)) # ~49
L6 = Int(ceil(L5/2)) # ~25
L7 = 10

# Neural net architecture
function neural_net()
    nn = Chain(
        Dense(L1, L2, relu),
        Dense(L2, L3, relu),
        Dense(L3, L4, relu),
        Dense(L4, L5, relu),
        Dense(L5, L6, relu),
        Dense(L6, L7, relu),
        Dense(L7, 1, x->σ.(x))
        )
    return nn
end

# Makes DataLoader classes for test/train matrices
dl_test = Flux.Data.DataLoader((conv_test_mat', class_test'))
dl_train = Flux.Data.DataLoader((conv_train_mat', class_train'),
                                    batchsize = batchsize_custom, shuffle = true)

nn = neural_net()
opt = RADAM()
loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))


# Actual training
ps = Flux.params(nn)
    for i in 1:epochs
        println(i)
        Flux.train!(loss, ps, dl_train, opt)
    end

# Testing for accuracy (at the end)
temppreds = []
for (x,y) in dl_test
    push!(temppreds,nn(x)[1])
end

push!(trueValues, class_test)
push!(predictions,temppreds)
push!(errorrates, 1-(sum((temppreds .> .5) .== class_test)/size(class_test)[1]))
println("round ",length(errorrates) , ": ", round((errorrates[end] * 100), digits = 2), "%")

# --------------------- Plotting ---------------------

JLD.save("Preds" * string(j) * ".jld", "val", predictions)
JLD.save("Trues" * string(j) * ".jld", "val", trueValues)
JLD.save("Errors" * string(j) * ".jld", "val", errorrates)
