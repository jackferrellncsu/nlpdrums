include("DTMCreation.jl")
include("data_cleaning.jl")
include("Packages.jl")
using Knet

datatot = importClean()
sort!(datatot, "medical_specialty")
createCorpusText(datatot,10)

#defining the lengths of the syntanctic or topical embeddings
vecLength1 = 10
   vecLength2 = 5

#Defining the window sizes
window1 = 30
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


rm("corpus.txt")

datasub = filtration(datatot, field)
   data, test = TrainTestSplit(datasub, .9);

#The output vector
class = data[:,1] .== field

Scripts = []
   for i in 1:length(class)
            push!(Scripts,vcat(formulateTextRNN(M,data[i,3])))
   end

rn = Chain(Flux.RNN(10,5),Dense(5,1,x -> σ.(x)))

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

opt = Flux.ADAM()

epochs = 500
for i in 1:epochs
    println(i, " : ", loss(Scripts,class))
    Flux.train!(lossSig , ps , zip(Scripts,class) , opt)
end




function formulateTextRNN(model, script)
   words = split(script, " ")
   vecs = []
   for i in words[1:end]
      if i in vocabulary(model) && i ∉ stopwords(Languages.English())
         push!(vecs, convert(Vector{Float32},get_vector(model,i)))
      end
   end
   return vecs
end
