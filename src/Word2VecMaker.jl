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
data = importClean()
data, test = TrainTestSplit(data, .9);


M = makeVectors(data,50,10)

class = data[:,1] .== " Cardiovascular / Pulmonary"
vecs = Matrix{Float64}(undef,length(class),50)
for i in 1:length(class)
   vecs[i,:] = formulateText(M,data[i,3])
end

ar = hcat(vecs,class)

df = DataFrame(ar,:auto)

ii = 50
z=term(Symbol(:x, ii+1)) ~ sum(term.(Symbol.(names(df[:, Not(Symbol(:x, ii+1))]))))

logit = glm(z,df, Bernoulli(), LogitLink())

classtest = test[:,1] .== " Cardiovascular / Pulmonary"
vecstest = Matrix{Float64}(undef,length(classtest),50)
for i in 1:length(classtest)
   vecstest[i,:] = formulateText(M,test[i,3])
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

   return model

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

#=
word = "heart"
list = cosine_similar_words(model, word,20)
coslist = []
for i in list
   push!(coslist,Word2Vec.similarity(model,word,i))
end

Plots.bar(reverse(list),reverse(coslist), yticks = :all,
            orientation=:h,leg = false)
Plots.xlabel!("Cosine")

w1, w2, w3 = ["elbow","arm","leg"]
println(analogy_words(model, [w1, w3], [w2], 10))


norm(get_vector(model,"heart"))
=#
