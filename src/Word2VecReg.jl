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

word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true,
window = window1)
M = wordvectors("vectors.txt", normalize = false)

rm("vectors.txt")

#second vectors
word2vec("corpus.txt", "vectors.txt", size = vecLength2, verbose = true,
window = window2)
M2 = wordvectors("vectors.txt", normalize = false)

rm("vectors.txt")

errors = []
for n in 1:25
   println(n)

   #Creating sub data set with training and testing
   datasub = filtration(datatot, field)
   data, test = TrainTestSplit(datasub, .9);


   class = data[:,1] .== field

   #Concatination
   vecs = zeros(length(class),vecLength1 + vecLength2)
   for i in 1:length(class)
         vecs[i,:] = vcat(formulateText(M,data[i,3]),formulateText(M2,data[i,3]))
   end

   df = DataFrame(hcat(vecs,class),:auto)

   #Fitting model
   ii = vecLength1 + vecLength2
   z=term(Symbol(:x, ii+1)) ~ sum(term.(Symbol.(names(df[:, Not(Symbol(:x, ii+1))]))))
   logit = glm(z,df, Bernoulli(), LogitLink())

   classtest = test[:,1] .== field
   vecstest = Matrix{Float64}(undef,length(classtest),ii)
   for i in 1:length(classtest)
         vecstest[i,:] = vcat(formulateText(M,test[i,3]),formulateText(M2,test[i,3]))
   end

   #Calculating Error Rate
   artest = hcat(vecstest,classtest)

   dftest = DataFrame(artest,:auto)

   preds = GLM.predict(logit,dftest)

   rez = preds.>.5

   #Adding to error rate
   push!(errors, 1- sum(rez.==classtest)/length(classtest))

end

trap

function formulateText(model, script)
   words = split(script, " ")
   vecs = zeros(length(get_vector(model,"the")))
   counter = 0
   for i in words[1:end]
      if i in vocabulary(model) && i ∉ stopwords(Languages.English())
         vecs = vecs .+ get_vector(model,i)
          #&& i ∉ stopwords(Languages.English())
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
      if uni[i] in vocabulary(model) && uni[i] ∉ stopwords(Languages.English())
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

function mostImportantWord(M,script,num)
   uni = unique(split(script," "))
   names = []
   norms = []
   coolnames = []
   for i in 1:length(uni)
      if uni[i] in vocabulary(model) && uni[i] ∉ stopwords(Languages.English())
         push!(norms, norm(get_vector(model,uni[i])))
         push!(names,uni[i])
      end
   end
   for i in 1:num
      ind = argmax(norms)
      push!(coolnames, names[ind])
      norms[ind] = 0
   end
   return coolnames
end

function filtration(df, field)
   indexes = []
   for i in 1:length(df[:,1])
      if df[i,:] == field
         push!(indexes,i)
      else
         if rand() < sum(df[:,1].==field)/(length(df[:,1]) - sum(df[:,1].==field))
            push!(indexes,i)
         end
      end
   end
   return df[indexes,:]
end

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
