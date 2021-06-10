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
using Lathe.preprocess: TrainTestSplit
using LinearAlgebra

data = importClean()
data, test = TrainTestSplit(data, .9);

vecLength = 10
window = 10
keeps = 10

T = makeVectors(data,vecLength,window)

M = T[1]
AD = T[2]

class = data[:,1] .== " Cardiovascular / Pulmonary"

vecs = Matrix{Float64}(undef,length(class),vecLength*keeps)
for i in 1:length(class)
   vecs[i,:] = formulateBest(M,data[i,3],keeps)
end

df = DataFrame(hcat(vecs,class),:auto)

ii = vecLength*keeps
z=term(Symbol(:x, ii+1)) ~ sum(term.(Symbol.(names(df[:, Not(Symbol(:x, ii+1))]))))
logit = glm(z,df, Bernoulli(), LogitLink())

classtest = test[:,1] .== " Cardiovascular / Pulmonary"
vecstest = Matrix{Float64}(undef,length(classtest),50)
for i in 1:length(classtest)
   vecstest[i,:] = formulateBest(M,test[i,3],keeps)
end

artest = hcat(vecstest,classtest)

dftest = DataFrame(artest,:auto)

preds = GLM.predict(logit,dftest)

rez = preds.>.5

1- sum(rez.==classtest)/length(classtest)

function makeVectors(data,vecLength, windowLength)
   allDocs = ""
   Padding = " afbjvaavdakjfbvavafvbasfdvlkajsvb"
   for i in 1:windowLength
      Padding = Padding * Padding
   end
   Padding = Padding * " "
   for i in data[:,3]
       allDocs = allDocs * Padding * i
   end

   open("corpus.txt","a") do io
      println(io,allDocs)
   end;

   word2vec("corpus.txt", "vectors.txt", size = vecLength, verbose = true,
    window = windowLength)

   model = wordvectors("vectors.txt", normalize = false)

   return [model, allDocs]

end

norm(get_vector(model,"heart"))

function formulateText(model, script)
   words = split(script, " ")
   vecs = zeros(length(get_vector(model,"the")))
   counter = 0
   for i in words[1:end]
      if i in vocabulary(model)
         vecs = vecs .+ get_vector(model,i)
         counter += 1
      end
   end
   return vecs ./ counter
end


function formulateBest(model,script, keeps)
   uni = unique(split(script," "))
   names = []
   vecs = zeros(length(uni),length(get_vector(model,"the")))
   norms = []
   bigvec = zeros(length(get_vector(model,"the")),keeps)
   for i in 1:length(uni)
      if uni[i] in vocabulary(model)
         vecs[i,:] = get_vector(model,String(uni[i]))
         push!(norms, norm(get_vector(model,uni[i])))
         push!(names,uni[i])
      end
   end
   for i in 1:keeps
      ind = argmax(norms)
      bigvec[:,i] = vecs[ind,:]'
      norms[ind] = 0
   end
   return vec(reshape(bigvec,1,keeps*length(get_vector(model,"the"))))
end
