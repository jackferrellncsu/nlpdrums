include("DTMCreation.jl")
include("data_cleaning.jl")
using Word2Vec
using MultivariateStats
using Lathe
using DataFrames
using Plots
using GLM
using StatsBase
using MLBase
using CSV
using Statistics
using Languages
using Lathe.preprocess: TrainTestSplit
using LinearAlgebra
using Clustering

datatot = importClean()
sort!(datatot, "medical_specialty")
rm("corpus.txt")
createCorpusText(datatot,10)

vecLength1 = 5
vecLength2 = 15
#Max is 50
window1 = 5
window2 = 30

field = " Cardiovascular / Pulmonary"

word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true, window = window1)
M = wordvectors("vectors.txt", normalize = true)

rm("vectors.txt")

#second vectors
word2vec("corpus.txt", "vectors.txt", size = vecLength2, verbose = true,window = window2)
M2 = wordvectors("vectors.txt", normalize = true)

rm("vectors.txt")

cdat = zeros(length(vocabulary(M)[2:end]),vecLength1+vecLength2)
for i in 1:length(vocabulary(M)[2:end])
    cdat[i,:] = vcat(get_vector(M,vocabulary(M)[i+1]),get_vector(M2,vocabulary(M2)[i+1]))
end

R = kmeans(cdat', 10, display=:iter)

function createCorpusText(data,pads)
   allDocs = ""
   #=
   Padding = " aaaa"
   for i in 1:pads
      println(i)
      Padding = Padding * Padding
   end
   Padding = Padding * " "
   =#
   for i in 1:length(data[:,3])
      println(i)
      allDocs = allDocs * " " * data[i,3]
   end

   open("corpus.txt","a") do io
      println(io,allDocs)
   end;
end

assign = Dict{String,Any}()


   for i in 1:length(vocabulary(M)[2:end])
      merge!(assign,Dict(vocabulary(M)[i+1]=>assignments(R)[i]))
   end
