include("../DTMCreation.jl")
include("../data_cleaning.jl")
include("Packages.jl")
include("../embeddings_nn.jl")


datatot = importCleanSentances()
sort!(datatot, "medical_specialty")
createCorpusText(datatot,1)

#defining the lengths of the syntanctic or topical embeddings
vecLength1 = 5

#Defining the window sizes
window1 = 500

#Creating the syntactic vector
word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true,
            window = window1, negative = 10, min_count = 0)
   M = wordvectors("vectors.txt", normalize = true)

   rm("vectors.txt")

   #creting the topical vectors

#Desired field
field = " Cardiovascular / Pulmonary"


rm("corpus.txt")

datasub = filtration(datatot, field)
   data, test = TrainTestSplit(datasub, .9);

#The output vector
class = data[:,1] .== field

Scripts = []
for i in 1:length(class)
            push!(Scripts,vcat(formulateTextRNN(M,data[i,3],.5)))
   end

rn = Chain(Flux.LSTM(25,1), x->σ.(x))

#rn = Flux.RNN(5,1,x -> σ.(x))

ps = Flux.params(rn)

function lossSig(x,y)
    ypred = rn.(x)[end][1]
    l = y*(-log(ypred)) + (1-y)*(-log(1-ypred))
    Flux.reset!(rn)
    return l
end

function loss(X,Y)
    sum = 0
    for i in 1:length(Y)
        sum += lossSig(X[i],Y[i])
    end
    return sum
end

#lossSig(x,y) = Flux.Losses.crossentropy(rn.(x)[end][1],y)

opt = ADAM()

epochs = 1000
Keeps = []
err = []
for a in 1:epochs
    while isnan(ps[1][1]) == false
        push!(Keeps,deepcopy(ps))
        Flux.train!(lossSig , ps , zip(Scripts,class) , opt)
        e = loss(Scripts,class)
        push!(err,e)
        println(" : ", e)
    end
end

classtest = test[:,1] .== field

Scriptstest = []
   for i in 1:length(classtest)
            push!(Scriptstest,vcat(formulateTextRNN(M,data[i,3],1)))
   end

correct = 0
for i in 1:length(classtest)
   correct += ((rn.(Scriptstest[i])[end][1] .> .5) .== classtest[i])
end
println(correct/length(classtest))

correct = 0
for i in 1:length(class)
   correct += ((rn.(Scripts[i])[end][1] > .5) .== class[i])
end

println(correct/length(class))



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

function formulateTextRNNSentances(model, script)
   vecs = []
   sentances = split(script, ".")
   for sentances in sentances
      subvecs = []
      words = split(sentances, " ")
      for word in words
         if word in vocabulary(model)
            push!(subvecs, convert(Vector{Float32},get_vector(model,word)))
         end
      end
      push!(vecs,subvecs)
   end
   return vecs
end
