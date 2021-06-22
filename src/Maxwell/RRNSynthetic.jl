using Pkg

Pkg.add("Word2Vec")
Pkg.add("CSV")
Pkg.add("Lathe")
Pkg.add("LinearAlgebra")
Pkg.add("DataFrames")
Pkg.add("Statistics")
Pkg.add("Flux")
Pkg.add("Plots")
Pkg.add("Languages")
Pkg.add("JLD")
Pkg.add("Random")

using Word2Vec
using Lathe
using DataFrames
using Plots
using CSV
using Statistics
using Languages
using Lathe.preprocess: TrainTestSplit
using LinearAlgebra
using Flux
using Tables
using JLD
using Random

function formulateTextRNN(model, script, prop)
   words = split(script, " ")
   vecs = []
   for i in words[1:end]
      if (i in vocabulary(model) && length(vecs) == 0 #=&& i ∉ stopwords(Languages.English())=#)||(i in vocabulary(model) #=&& i ∉ stopwords(Languages.English())=# && rand() < prop)
         push!(vecs, convert(Vector{Float32},get_vector(model,i)))
      end
   end
   return vecs
end

datatot = CSV.read("wordy.csv", DataFrame)

#defining the lengths of the syntanctic or topical embeddings
vecLength1 = 15

#Defining the window sizes
window1 = 50
#Creating the syntactic vector

M = wordvectors("ForeignVectors.txt", normalize = true)

#Desired field

errorrates = []
predictions = []
trueValues = []

for i in 1:1000
   Random.seed!(i)
   data, test = TrainTestSplit(datatot[(i-1)*1000+1:(i)*1000,:], .9);

   Scripts = []
   for i in 1:size(data)[1]
               push!(Scripts,(formulateTextRNN(M,data[i,1],1)))
      end

   rn = Chain(Flux.GRU(15,10),Dense(10,10,swish),Dense(10,1, x->σ.(x)))

   #rn = Flux.RNN(5,1,x -> σ.(x))

   ps = Flux.params(rn)


   function loss(x, y)
      if rand() <= .1
         Flux.reset!(rn)
         return sum(Flux.Losses.binarycrossentropy(rn.(x)[end], y))
      else
         return 0
      end
   end


   opt = RADAM()

   epochs = 50

   for i in 1:epochs
      println(i)
      Flux.train!(loss , ps , zip(Scripts,data[:,2]) , opt)
   end


   Scriptstest = []
      for i in 1:size(test)[1]
            push!(Scriptstest,(formulateTextRNN(M,test[i,1],1)))
      end

   temppreds = []
   for i in 1:size(test)[1]
      Flux.reset!(rn)
      push!(temppreds,rn.(Scriptstest[i])[end][1])
   end

   push!(trueValues, test[:,2])
   push!(predictions,temppreds)
   push!(errorrates, 1-(sum((temppreds .> .5) .== test[:,2])/size(test)[1]))
   println("round:",length(errorrates) , " : ", errorrates[end])

end

rm("wordy.csv")

JLD.save("RNNpredictions.jld", "predictions", predictions)
JLD.save("RNNtrueValues.jld", "trueValues", trueValues)
JLD.save("RRNerrorrates.jld", "errorrates", errorrates)
