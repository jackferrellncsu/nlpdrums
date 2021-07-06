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
# ====================== Variables ====================== #

trainTestSplitPercent = .7
batchsize_custom = 100
epochs = 500

errorrates = []
predictions = []
trueValues = []

# ====================== Conv Loop ===================== #

# Reads job number and labels it "j"
j = parse(Int64, get(parsed_args, "arg1", 0 ))

# Pulls values from job that are necessary
parameters_conv = [05, 10, 15, 20, 25, 30]
parameters_pool = [200, 300, 400, 500, 600, 700]
parameters = []
for i in 1:length(parameters_conv)
    for j in 1:length(parameters_pool)
        if parameters_conv[i] == 5
            new = "0" * string(parameters_conv[i]) * string(parameters_pool[j])
            push!(parameters, new)
        else
            new = string(parameters_conv[i]) * string(parameters_pool[j])
            push!(parameters, new)
        end
    end
end
seed = j % 100
Random.seed!(seed)
par = Int(ceil(j/100))
param_conv_string = parameters[par][1] *  parameters[par][2]
param_pool_string = parameters[par][3] *  parameters[par][4] *  parameters[par][5]
param_conv = parse(Int64, param_conv_string)
param_pool = parse(Int64, param_pool_string)

# ******* FILENAME FOR TESTING ******* #
#filename = "/Users/eplanch/Documents/GitHub/nlpdrums/src/JonsTraining.csv"

# ******* FILENAME FOR CLUSTER ******* #
filename = "JonsTraining.csv"

true_data = CSV.read(filename, DataFrame)
DTM = CreateDTM(true_data)
println("DTM Done")

# Create dataframe for subset
total_DTM = DataFrame(DTM')

# Split into train and test (validation) set
train, test = TrainTestSplit(total_DTM, trainTestSplitPercent)

# Finding classifcation vectors
class_train = train[:, end]
class_test = test[:, end]

# Removing classification columns
dtm_train = Matrix(train[:, 1:end-1])
dtm_test = Matrix(test[:, 1:end-1])

########################## Beginning of Convolution ##########################

# Convolutional Layer
num_rows_train = length(dtm_train[:,1])
num_rows_test = length(dtm_test[:,1])
layers_train = Chain(
                Conv(tuple(1, param_conv), 1 => 1, relu),
                AdaptiveMaxPool(tuple(num_rows_train, param_pool)))
layers_test = Chain(
                Conv(tuple(1, param_conv), 1 => 1, relu),
                AdaptiveMaxPool(tuple(num_rows_test, param_pool)))
println("Start Conv")

# Convolution & Pooling for training matrix
train1 = length(dtm_train[:,1])
train2 = length(dtm_train[1,:])
train_array = reshape(dtm_train, (train1, train2, 1, 1))
conv_train_array = layers_train(train_array)
conv_train_mat = conv_train_array[1, :]'
for i in 2:length(conv_train_array[:,1])
    global conv_train_mat = vcat(conv_train_mat, conv_train_array[i,:]')
end

# Convolution & Pooling for testing matrix
test1 = length(dtm_test[:,1])
test2 = length(dtm_test[1,:])
test_array = reshape(dtm_test, (test1, test2, 1, 1))
conv_test_array = layers_test(test_array)
conv_test_mat = conv_test_array[1, :]'
for i in 2:length(conv_test_array[:,1])
    global conv_test_mat = vcat(conv_test_mat, conv_test_array[i,:]')
end

# Making layers for neural net
L1 = length(conv_test_mat[1,:])
L2 = Int(ceil(L1/3))
L3 = Int(ceil(L2/3))
L4 = Int(ceil(L3/2))
L5 = Int(ceil(L4/2))

# Neural net architecture
function neural_net()
    nn = Chain(
        Dense(L1, L2, relu),
        Dense(L2, L3, relu),
        Dense(L3, L4, relu),
        Dense(L4, L5, relu),
        Dense(L5, 1, x->σ.(x))
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
println("Start Train")

# Actual training
ps = Flux.params(nn)
    for i in 1:epochs
        println(string(i))
        Flux.train!(loss, ps, dl_train, opt)
    end

# Testing for accuracy (at the end)
temppreds = []
for (x,y) in dl_test
    push!(temppreds,nn(x)[1])
end

push!(errorrates, 1-(sum((temppreds .> .5) .== class_test)/size(class_test)[1]))

println("Round ",length(errorrates) , ": ", round((errorrates[end] * 100), digits = 2), "%")

JLD.save("cv_errors" * string(j) * ".jld", "val", errorrates)
