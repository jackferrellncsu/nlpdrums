include("../DTMCreation.jl")
include("../data_cleaning.jl")
include("Packages.jl")
include("../embeddings_nn.jl")


datatot = importClean()
sort!(datatot, "medical_specialty")
createCorpusText(datatot,1)

#defining the lengths of the syntanctic or topical embeddings
vecLength1 = 25

#Defining the window sizes
window1 = 500

#Creating the syntactic vector
word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true,
            window = window1, negative = 10, min_count = 0)
   M = wordvectors("vectors.txt", normalize = true)

   rm("vectors.txt")

   #creting the topical vectors

rm("corpus.txt")

data, test = TrainTestSplit(datatot, .9);

fieldClass = []
uniqueFields = unique(data[:, 1])
for i in 1:length(data[:,1])
    push!(fieldClass, Flux.onehot(data[i, 1], uniqueFields))
end

Scripts = []
for i in 1:length(fieldClass)
            push!(Scripts,formulateTextRNN(M,data[i,3],.5))
   end

rn = Chain(Flux.RNN(25,29, swish), Dense(29,29,x->σ.(x)))

ps = Flux.params(rn)

function loss(x, y)
   Flux.reset!(rn)
   return sum(Flux.Losses.logitcrossentropy(rn.(x)[end], y))
end

opt = Descent(.01)

epochs = 1000
Keeps = []
err = []
for i in 1:epochs
    while isnan(ps[1][1]) == false
        push!(Keeps,deepcopy(ps))
        Flux.train!(loss , ps , zip(Scripts,fieldClass) , opt)
        push!(err,sum(loss.(Scripts,fieldClass)))
        print(Keeps[end])
    end
end


function formulateTextRNN(model, script, prop)
      words = split(script, " ")
      vecs = []
      for i in words[1:end]
         if (i in vocabulary(model) && length(vecs) == 0 && i ∉ stopwords(Languages.English()))||(i in vocabulary(model) && i ∉ stopwords(Languages.English()) && rand() < prop)
            push!(vecs, convert(Vector{Float32},get_vector(model,i)))
         end
      end
      return vecs
end

Flux.loadparams!(rn,Keeps[end])
