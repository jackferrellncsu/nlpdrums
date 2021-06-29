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

using Word2Vec
using DataFrames
using GLM
using StatsBase
using CSV
using Statistics
using Languages
using Lathe.preprocess: TrainTestSplit
using LinearAlgebra
using JLD
using Random
using Flux

vecLength1 = 15
vecLength2 = 0
#Max is 50
window1 = 50

function createCorpusText(data, choice)
   allDocs = ""
   thePad = ""
   for i in 1:3000
      thePad = thePad * " randomWordNow"
   end
   for i in 1:length(data[:, 3])
      println(i)
      if choice == 1
         if i != 1
            if data[i, 1] != data[i-1, 1]
               println("This is a seperation")
               allDocs = allDocs * thePad * " " * data[i, 3]
            else
               allDocs = allDocs * " " * data[i, 3]
            end
         end
      elseif choice == 0
         allDocs = allDocs * " " * data[i, 3]
      elseif choice == 2
         allDocs = allDocs * thePad * " " * data[i, 3]
      end
   end
   open("corpus.txt","a") do io
      println(io,allDocs)
   end
end

function importClean()
    filename = "cleanedData.csv"
    filepath = joinpath(@__DIR__, filename)
    arr = CSV.read(filepath, DataFrame)

    return arr
end

function filtration(df, field)
   indexes = []
   for i in 1:length(df[:,1])
      if df[i,1] == field
         push!(indexes,i)
      else
         if rand() < sum(df[:,1].==field)/(length(df[:,1]) - sum(df[:,1].==field))
            push!(indexes,i)
         end
      end
   end
   return df[indexes,:]
end

function formulateText(model, script)
   words = split(script, " ")
   vecs = zeros(length(get_vector(model,"the")))
   for i in words[1:end]
      if i in vocabulary(model) && i ∉ stopwords(Languages.English())
         vecs = vecs .+ get_vector(model,i)
      end
   end
   return vecs
end

JLD.save("PredsFinal.jld", "run0", 0)
JLD.save("TruesFinal.jld", "run0", 0)
JLD.save("ErrorsFinal.jld", "run0", 0)
for n in 1:50
   println("----------------" , n)
   field = " Cardiovascular / Pulmonary"
   Random.seed!(n)
   D = CSV.read("/Users/mlovig/Documents/GitHub/nlpdrums/src/cleanedData.csv", DataFrame)
   Random.seed!(n)
   train, test = TrainTestSplit(D, .9);

   Random.seed!(n)
   data = filtration(train, " Cardiovascular / Pulmonary")
   createCorpusText(train,0)

   #defining the lengths of the syntanctic or topical embeddings
   vecLength1 = 15

   #Defining the window sizes
   window1 = 50

   #Creating the syntactic vector
   word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true, window = window1, negative = 10, min_count = 0)
      M = wordvectors("vectors.txt", normalize = true)

      rm("vectors.txt")
      #creting the topical vectors

   rm("corpus.txt")

   vecs = zeros(size(data)[1],vecLength1)
   for i in 1:size(data)[1]
       vecs[i,:] = formulateText(M,data[i,1])
   end

   classtest = test[:,1] .== field
   ii = vecLength1 + vecLength2
   vecstest = Matrix{Float64}(undef,length(classtest),ii)
   for i in 1:length(classtest)
      vecstest[i,:] = formulateText(M,test[i,1])
   end

   classtest = test[:,1] .== field
   class = data[:,1] .== field

   function neural_net()
      nn = Chain(
           Dense(15, 30, swish),Dense(30, 10, swish),
           Dense(10, 1, x->σ.(x))
           )
       return nn
   end

   # Makes "DataLoader" classes for both testing and training data
   # Batchsize for training shoulde be < ~size(train). Way less
   newTestData = Flux.Data.DataLoader((vecstest', classtest'))
   newTrainingData = Flux.Data.DataLoader((vecs', class'), shuffle = true, batchsize = 100)

   # Defining our model, optimization algorithm and loss function
   # @function Descent - gradient descent optimiser with learning rate η
   nn = neural_net()
   opt = RADAM()
   ps = Flux.params(nn)
   loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))

   # Actual training
   traceY = []
   traceY2 = []
   epochs = 500
       for i in 1:epochs
           Flux.train!(loss, ps, newTrainingData, opt)

            println(i)
           #=
           for (x,y) in trainDL
               totalLoss = loss(x,y)
           end
           push!(traceY, totalLoss)
           =#
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
   preds = []
   trues = []
       for (x,y) in newTestData
           push!(preds, nn(x)[1])
           push!(trues,y[1])
       end

   errors = 1 - sum((preds .> .5) .== trues)/length(trues)

    jldopen("PredsFinal.jld", "r+") do file
         write(file, "run"*string(n), preds)  # alternatively, say "@write file A"
    end
    jldopen("TruesFinal.jld", "r+") do file
         write(file, "run"*string(n), trues)  # alternatively, say "@write file A"
    end
    jldopen("ErrorsFinal.jld", "r+") do file
         write(file,"run"*string(n), errors)  # alternatively, say "@write file A"
    end

end
