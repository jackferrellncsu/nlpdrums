include("../DTMCreation.jl")
include("../data_cleaning.jl")
include("Packages.jl")
include("../emi/embeddingsForMax.jl")


datatot = CSV.read("wordy.csv", DataFrame)
createCorpusText(datatot,0)

#defining the lengths of the syntanctic or topical embeddings
vecLength1 = 5

#Defining the window sizes
window1 = 50

#Creating the syntactic vector
word2vec("corpus.txt", "vectors.txt", size = vecLength1, verbose = true,
            window = window1, negative = 10, min_count = 0)
   M = wordvectors("vectors.txt", normalize = true)

   rm("vectors.txt")

   #creting the topical vectors

#Desired field

rm("corpus.txt")


data, test = TrainTestSplit(datatot, .9);

Scripts = []
for i in 1:size(data)[1]
            push!(Scripts,(formulateTextRNN(M,data[i,1],1)))
   end

rn = Chain(Flux.LSTM(5,10),Dense(10,1, x->σ.(x)))

#rn = Flux.RNN(5,1,x -> σ.(x))

ps = Flux.params(rn)


function loss(x, y)
   if rand() <= .66
      Flux.reset!(rn)
      return sum(Flux.Losses.binarycrossentropy(rn.(x)[end], y))
   else
      return 0
   end
end


opt = RADAM()

epochs = 1000
Keeps = []
err = []
for i in 1:1000
    while isnan(ps[1][1]) == false
        push!(Keeps,deepcopy(ps))
        Flux.train!(loss , ps , zip(Scripts,data[:,2]) , opt)
        e = sum(loss.(Scripts,data[:,2]))
        push!(err,e)
        print(i, " : ", e)
    end
end


Scriptstest = []
   for i in 1:size(test)[1]
         push!(Scriptstest,(formulateTextRNN(M,test[i,1],1)))
   end

correct = 0
for i in 1:size(test)[1]
   Flux.reset!(rn)
   correct += ((rn.(Scriptstest[i])[end][1] .> .5) .== test[i,2])
end
println(1-correct/size(test)[1])

correct = 0
for i in 1:size(data)[1]
   correct += ((rn.(Scripts[i])[end][1] > .5) .== data[i,2])
end

println(correct/size(data)[1])



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
