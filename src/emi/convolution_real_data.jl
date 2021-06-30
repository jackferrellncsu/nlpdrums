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


include("PCA.jl")
include("data_cleaning.jl")
include("embeddings_nn.jl")
include("DTMCreation.jl")
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
    true_data = importClean()
    sort!(true_data, "medical_specialty")

    # Clean data and test/train split
    # Creates DTM (train & test)
    field = " Cardiovascular / Pulmonary"
    DTM = CreateDTM(true_data, field)
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
            MaxPool(tuple(1, 70)))

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

    # Neural net architecture
    function neural_net()
        nn = Chain(
            Dense(327, 200, relu),
            Dense(200, 125, relu),
            Dense(125, 75, relu),
            Dense(75, 45, relu),
            Dense(45, 20, relu),
            Dense(20, 8, relu),
            Dense(8, 1, x->σ.(x))
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
            Flux.train!(loss, ps, dl_train, opt)
        end

    # Saving model predictions
    temppreds = []
    for (x,y) in dl_test
            push!(temppreds,nn(x)[1])
    end


    # Saving model true values and error rates
    push!(trueValues, class_test)
    push!(predictions,temppreds)
    push!(errorrates, 1-(sum((temppreds .> .5) .== class_test)/size(class_test)[1]))
    println("round ",length(errorrates) , ": ", round((errorrates[end] * 100), digits = 2), "%")
    # --------------------- Plotting ---------------------


JLD.save("Preds" * string(j) * ".jld", "val", predictions)
JLD.save("Trues" * string(j) * ".jld", "val", trueValues)
JLD.save("Errors" * string(j) * ".jld", "val", errorrates)
