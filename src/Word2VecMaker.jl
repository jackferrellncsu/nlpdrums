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


#T = makeVectors(data,50,10)

word2vec("text8", "text8-vec.txt", verbose = true, window = 20)
M = wordvectors("text8-vec.txt", normalize = false)

io = open("text8", "r");
AD = read(io, String)
#=
M = T[1]
AD = T[2]
=#
vecnorms = []
occurs = []
counter = 0
cm = countmap(split(AD, " "))
for i in vocabulary(M)[2:end]
   counter += 1
   println(counter)
   push!(vecnorms, norm(get_vector(M,i)))
   push!(occurs,cm[i])
end


vecnorms1 = []
counter = 0
for i in vocabulary(M)
   counter += 1
   println(counter)
   push!(vecnorms1, sum(abs.(get_vector(M,i))))
end

vecnormsmax = []
counter = 0
for i in vocabulary(M)
   counter += 1
   println(counter)
   push!(vecnormsmax, maximum(get_vector(M,i)))
end


p1 = plot(log10.(occurs),vecnorms, seriestype = :scatter, leg = false)
xlabel!("Log(Occurances)")
ylabel!("Norm2(Vector)")

p2 = plot(log.(occurs),vecnorms1, seriestype = :scatter, leg = false)
xlabel!("Log(Occurances)")
ylabel!("Norm1(Vector)")

p3 = plot(log.(occurs),vecnormsmax, seriestype = :scatter, leg = false)
xlabel!("Log(Occurances)")
ylabel!("NormMax(Vector)")

plot(p1,p2,p3)

veced = copy(vecnorms)
vocab = copy(vocabulary(M))
impvecs = []
impvocab = []
for i in 1:100
   ind = argmax(veced)
   push!(impvecs, veced[ind])
   push!(impvocab, vocab[ind])
   veced = vcat(veced[1:ind-1] ,veced[ind+1:end])
   vocab = vcat(vocab[1:ind-1] ,vocab[ind+1:end])
end
#=
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
=#
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
