include("../DTMCreation.jl")
include("../data_cleaning.jl")
include("Packages.jl")
include("../embeddings_nn.jl")


datatot = importCleanSentances()
sort!(datatot, "medical_specialty")
createCorpusText(datatot,1)

#defining the lengths of the syntanctic or topical embeddings
vecLength1 = 30

#Defining the window sizes
window1 = 30

#Creating the syntactic vector
word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true,
            window = window1, negative = 10, min_count = 0)
   M = wordvectors("vectors.txt", normalize = true)

   rm("vectors.txt")

   #creting the topical vectors

#Desired field
field = " Cardiovascular / Pulmonary"


rm("corpus.txt")

num_clusters = 25

datasub = filtration(datatot, field)
   data, test = TrainTestSplit(datasub, .8);

   cdat = zeros(length(vocabulary(M)[1:end]),vecLength1)
   for i in 1:length(vocabulary(M)[1:end])
       cdat[i,:] = vcat(get_vector(M,vocabulary(M)[i]))
   end

   R = kmeans(cdat', num_clusters, display=:iter)

   assign = Dict{String,Any}()


      for i in 1:length(vocabulary(M)[1:end])
         merge!(assign,Dict(vocabulary(M)[i]=>assignments(R)[i]))
      end


class = []
sents = []
for i in 1:length(data[:,1])
      v = formulateTextRNNSentancesCluster(M,data[i,3],20, R, num_clusters)
      append!(sents,v)
      for ii in 1:length(v)
         push!(class, data[i,1] .== field)
      end
end

<<<<<<< Updated upstream
rn = Chain(Flux.LSTM(25,10),Dense(10,1,x->σ.(x)))
=======
rn = Flux.RNN(25,1, sigmoid)
>>>>>>> Stashed changes

#rn = Flux.RNN(5,1,x -> σ.(x))

ps = Flux.params(rn)

opt = ADAM()

epochs = 1000
Keeps = []
err = []
for a in 1:epochs
    while isnan(ps[1][1]) == false
        push!(Keeps,deepcopy(ps))
        Flux.train!(lossSig , ps , zip(sents,class) , opt)
        e = loss(sents,class)
        push!(err,e)
        println(" : ", e)
    end
end

classtest = []
sentstest = []
for i in 1:length(test[:,1])
   v = formulateTextRNNSentancesCluster(M,test[i,3],10,R,num_clusters)
   append!(sentstest,v)
   for ii in 1:length(v)
      push!(classtest, test[i,1] .== field)
   end
end

correct = 0
for i in 1:length(classtest)
   correct += ((rn.(sentstest[i])[end][1] .> .5) .== classtest[i])
end
println(correct/length(classtest))

correct = 0
for i in 1:length(class)
   correct += ((rn.(sents[i])[end][1] > .5) .== class[i])
end
println(correct/length(class))

function formulateTextRNNSentances(model, script, minsents)
   vecs = []
   sentances = split(script, ".")

   for sentance in sentances
      subvecs = []
      words = split(sentance, " ")
      for word in words
         if word in vocabulary(model)
            push!(subvecs, convert(Vector{Float32},get_vector(model,word)))
         end
      end
      if length(subvecs)>minsents
         push!(vecs,subvecs)
      end
   end
   return vecs
end

function formulateTextRNNSentancesCluster(model, script, minsents, cluster, totclusters)
   vecs = []
   sentances = split(script, ".")

   for sentance in sentances
      subvecs = []
      words = split(sentance, " ")
      for word in words
         if word in vocabulary(model)
            V = zeros(totclusters)
            V[get(assign, String(word),1)] = 1
            push!(subvecs, convert(Vector{Float32},V))
         end
      end
      if length(subvecs)>minsents
         push!(vecs,subvecs)
      end
   end
   return vecs
end

function lossSig(x,y)
   if rand() <1
       ypred = rn.(x)[end][1]
       l = y*(-log(ypred)) + (1-y)*(-log(1-ypred))
       Flux.reset!(rn)
       return l
   end
   return 0
end

function loss(X,Y)
    sum = 0
    for i in 1:length(Y)
        sum += lossSig(X[i],Y[i])
    end
    return sum
end
