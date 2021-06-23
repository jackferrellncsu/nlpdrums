Pkg.add("Flux")
Pkg.add("using LinearAlgebra")
Pkg.add("using Statistics")
Pkg.add("using ROCAnalysis")
Pkg.add("using MLBase")
Pkg.add("using Plots")
Pkg.add("using RecursiveArrayTools")
Pkg.add("TextAnalysis")
Pkg.add("InvertedIndices")
Pkg.add("StatsBase")
Pkg.add("Random")

using Flux
using LinearAlgebra
using Statistics
using ROCAnalysis
using MLBase
using Plots
using RecursiveArrayTools
using TextAnalysis
using InvertedIndices
using StatsBase
using Random

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
        if author(i) == "1.0"
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
η = 0.01
epochs = 1000

errorrates = []
predictions = []
trueValues = []

# ====================================================== #
# ==================== Conv For Loop =================== #
# ====================================================== #
for i in 1:1000
    println("Iteration number: "*string(i))
    bVal = 1 + ((i * 1000) - 1000)
    eVal = 1000 * i
    # Imports clean data
    true_data = CSV.read("wordy.csv", DataFrame)
    true_data = true_data[bVal:eVal, :]

    # Create DTM
    DTM = CreateDTM(true_data)
    total_DTM = DataFrame(DTM')


    Random.seed!(i)
    train, test = TrainTestSplit(total_DTM, trainTestSplitPercent)


    # Finding classifcation vectors
    class_train = train[:, end]
    class_test = test[:, end]

    # Removing classification columns
    dtm_train = Matrix(train[:, 1:end-1])
    dtm_test = Matrix(test[:, 1:end-1])

    # Convolution & Pooling for training matrix
    train1 = length(dtm_train[:,1])
    train2 = length(dtm_train[1,:])
    train_array = reshape(dtm_train, (train1, train2, 1, 1))

    layers = Chain(
            # Manually reduces 10 (11 - 1) words off of the vocab
            Conv(tuple(1, 11), 1 => 1, relu),
            MaxPool(tuple(1, 2)))
    conv_train_array = layers(train_array)
    conv_train_mat = conv_train_array[1, :]'
    for i in 2:length(conv_train_array[:,1])
        conv_train_mat = vcat(conv_train_mat, conv_train_array[i,:]')
    end

    # Convolution & Pooling for testing matrix
    test1 = length(dtm_test[:,1])
    test2 = length(dtm_test[1,:])
    test_array = reshape(dtm_test, (test1, test2, 1, 1))
    layer = Chain(
            Conv(tuple(11, 1), 1 => 1, relu),
            MaxPool(tuple(2, 1)))
    conv_test_array = layers(test_array)
    conv_test_mat = conv_test_array[1, :]'
    for i in 2:length(conv_test_array[:,1])
        conv_test_mat = vcat(conv_test_mat, conv_test_array[i,:]')
    end

    # Neural net architecture
    function neural_net()
        nn = Chain(
            Dense(45, 30, relu),
            Dense(30, 15, relu),
            Dense(15, 7, relu),
            Dense(7, 1, x->σ.(x))
            )
        return nn
    end

    # Makes DataLoader classes for test/train matrices
    dl_test = Flux.Data.DataLoader((conv_test_mat', class_test'))
    dl_train = Flux.Data.DataLoader((conv_train_mat', class_train'),
                                        batchsize = batchsize_custom, shuffle = true)

    nn = neural_net()
    opt = Descent(η)
    loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))


    # Actual training
    ps = Flux.params(nn)
        for i in 1:epochs
            Flux.train!(loss, ps, dl_train, opt)
            println(i)
        end

    # Testing for accuracy (at the end)
    temppreds = []
    for (x,y) in dl_test
            push!(temppreds,nn(x)[1]) #%%%% ?? %%%#
    end

    push!(trueValues, class_test)
    push!(predictions,temppreds)
    push!(errorrates, 1-(sum((temppreds .> .5) .== class_test)/size(class_test)[1]))
    println("round ",length(errorrates) , ": ", round((errorrates[end] * 100), digits = 2), "%")
    # --------------------- Plotting ---------------------
end

JLD.save("CONVpredictions.jld", "predictions", predictions)
JLD.save("CONVtrueValues.jld", "trueValues", trueValues)
JLD.save("CONVerrorrates.jld", "errorrates", errorrates)
