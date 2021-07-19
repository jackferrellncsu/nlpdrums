using JLD
using Embeddings
using DataFrames
using Flux
using LinearAlgebra
using Statistics
using Random
using StatsBase
using BSON
using CUDA
using ProgressBars
using Plots

include("nextword_helpers.jl")

Random.seed!(26)

obj = load("PridePrej.jld");
    data = obj["data"];
    sentences = obj["sentances"];
    corpus = obj["corpus"];

embtable = load("pridePrejEmbs.jld", "embtable")

#get vector from word
get_vector_word = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab));
#get word from vector
get_word_vector = Dict(embtable.embeddings[:, ii] => word for (ii, word) in enumerate(embtable.vocab));
#get index from word
get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab));

vec_length = length(get_vector_word["the"]);

#filter out non-embedded outcomes
unique_words = [word for word in keys(get_vector_word)]
data = DataFrame(data)
filter!(row -> row[2] ∈ unique_words, data)
data = Matrix(data)


#Create input and output matrices
y_mat = BitArray(undef, 6575, 114798)

for i in 1:size(data)[1]
    y_mat[:, i]  = (Flux.onehot(data[i, 2], unique_words) .== 1)
end

x_mat = EmbeddingsTensor(data, get_vector_word)[1]

#Split input and output into propertrain/calibration/test
#split into test, proper_train, calibrate
train_x, test_x, train_y, test_y = SampleMats(x_mat, y_mat)
proper_train_x, calibrate_x, proper_train_y, calibrate_y = SampleMats(train_x, train_y, .92) |> gpu


#Put inputs and outputs into Flux Dataloader class for NN input
trainDL = Flux.Data.DataLoader((proper_train_x, proper_train_y),
                            batchsize = 1000,
                            shuffle = true)

calibrateDL = Flux.Data.DataLoader((calibrate_x, calibrate_y))
testDL = Flux.Data.DataLoader((test_x, test_y))

BSON.@load "softmod_cpu.bson" model

typeof(model)

trace = load("softmod_trace_gpu.jld", "trace")

for (x, y) in testDL
   ŷ = model(x)
   print(typeof(ŷ))
   break
end
