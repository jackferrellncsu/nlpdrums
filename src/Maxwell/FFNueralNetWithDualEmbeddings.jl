include("../DTMCreation.jl")
include("../data_cleaning.jl")
include("../embeddings_nn.jl")
include("clustering.jl")
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
using Flux
using Clustering


datatot = importClean()
sort!(datatot, "medical_specialty")
rm("corpus.txt")
createCorpusText(datatot,1)

#defining the lengths of the syntanctic or topical embeddings
vecLength1 = 20
   vecLength2 = 15

#Defining the window sizes
window1 = 500
   window2 = 30

#Creating the syntactic vector
word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true,
            window = window1)
   M = wordvectors("vectors.txt", normalize = true)

   rm("vectors.txt")

   #creting the topical vectors
   word2vec("corpus.txt", "vectors.txt", size = vecLength2, verbose = true,
            window = window2)
   M2 = wordvectors("vectors.txt", normalize = true)

   rm("vectors.txt")

#Desired field
field = " Cardiovascular / Pulmonary"

cdat = zeros(length(vocabulary(M)[2:end]),vecLength1)
for i in 1:length(vocabulary(M)[2:end])
    cdat[i,:] = #=vcat(=#get_vector(M,vocabulary(M)[i+1])#=,get_vector(M2,vocabulary(M2)[i+1]))=#
end

numassign = 15
R = kmeans(cdat', numassign, display=:iter)

assign = Dict{String,Any}()

for i in 1:length(vocabulary(M)[2:end])
   merge!(assign,Dict(vocabulary(M)[i+1]=>assignments(R)[i]))
end

#Creating sub data set with training and testing
datasub = filtration(datatot, field)
   data, test = TrainTestSplit(datasub, .75);

#The output vector
class = data[:,1] .== field

#Concatination

vecs = zeros(length(class),(vecLength1))
   for i in 1:length(class)
            vecs[i,:] = formulateText(M,data[i,3])

   end


#=
vecs = zeros(length(class),numassign)
   for i in 1:length(class)
            vecs[i,:] = formulateTextClusterSpec(M,data[i,3], assign,numassign)
   end
=#

#Repeating the same thing for the testing data

testclass = test[:,1] .== field

testvecs = zeros(length(testclass),(vecLength1))
   for i in 1:length(testclass)
            testvecs[i,:] = formulateText(M,test[i,3])

   end

#=
testvecs = zeros(length(testclass),numassign)
   for i in 1:length(testclass)
            testvecs[i,:] = formulateTextClusterSpec(M,test[i,3], assign,numassign)
   end
=#

#loading both testing a training into a DataLoader
d = Flux.Data.DataLoader((vecs',class'), batchsize=100, shuffle = true)

   dtest = Flux.Data.DataLoader((testvecs',testclass'), shuffle = false)

#Creating the Nueral Net, feed foward 20 -> 50 -> 1
nn = Chain(Dense(15,5, relu),Dense(5,1,sigmoid))

   opt = ADAM()

   ps = Flux.params(nn)

   loss(x,y) = sum(Flux.Losses.binarycrossentropy(nn(x),y))

#Training the nueral net
losses = []
epoch = 10000
   for i in 1:epoch
      e = loss(vecs',class')
      push!(losses,e)
      println(i, " : ", e)
      Flux.train!(loss, ps,d , opt)
   end

#Accessing the accuracy
acc = 0
   for (x,y) in dtest
      #=
      print(nn(x) .> .5, " : ")
      println(y)
      =#


      acc += (nn(x)[1]>.5) == y[1]
   end
   println(1- (acc/length(testclass)))

df = DataFrame(hcat(vecs,class),:auto)

dftest = DataFrame(hcat(testvecs,testclass),:auto)

ii = (vecLength1 + vecLength2)*numassign
z=term(Symbol(:x, ii+1)) ~ sum(term.(Symbol.(names(df[:, Not(Symbol(:x, ii+1))]))))
logit = glm(z,df, Bernoulli(), LogitLink())

acclogit = sum((GLM.predict(logit,dftest).>.5) .== testclass) / length(testclass)


preds = zeros(0)
for (x,y) in dtest

   #=
   print(nn(x) .> .5, " : ")
   println(y)
   =#


   push!(preds,nn(x)[1])
end






%---------------------------------------------------------
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
   return vecs
end

function formulateTextRNN(model, script)
   words = split(script, " ")
   vecs = zeros(length(get_vector(model,"the")))
   counter = 0
   for i in words[1:end]
      if i in vocabulary(model) && i ∉ stopwords(Languages.English())
         append!(vecs, get_vector(model,i))
      end
   end
   return vecs
end

function formulateTextCluster(model, script, assign, numassign)
   words = split(script, " ")
   vecs = zeros(length(get_vector(model,"the")), numassign)
   for i in words[1:end]
      if i in vocabulary(model) && i ∉ stopwords(Languages.English())
         vecs[:,get(assign, String(i),1)] = vecs[:,get(assign, String(i),1)] .+ get_vector(model,i)
          #&& i ∉ stopwords(Languages.English())
      end
   end
   return vecs
end

function formulateTextClusterSpec(model, script, assign, numassign)
   words = split(script, " ")
   vecs = zeros(numassign)
   for i in words[1:end]
      if i in vocabulary(model) && i ∉ stopwords(Languages.English())
         V = zeros(numassign)
         V[get(assign, String(i),1)] = 1
         vecs = vecs .+ convert(Vector{Float32},V)
          #&& i ∉ stopwords(Languages.English())
      end
   end
   return vecs
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
