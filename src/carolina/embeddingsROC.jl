using Lathe
using Flux
using Plots
using Word2Vec
using Lathe.preprocess: TrainTestSplit
using Languages
using MLBase
include("../data_cleaning.jl")

text = importClean()
sort!(text, "medical_specialty")
createCorpusText(text, 10)

field = " Cardiovascular / Pulmonary"
word2vec("corpus.txt", "vectors.txt", size = 15, window = 5)

m = wordvectors("vectors.txt", normalize = true)

dataS = filtration(text, field)

train, test = TrainTestSplit(dataS, 0.9)

classTrain = train[:,1].== field
classTest = test[:,1].== field

vecsTrain = zeros(length(classTrain), 15)
vecsTest = zeros(size(classTest)[1], 15)

for i in 1:length(classTrain)
    vecsTrain[i, :] = formulateText(m, train[i,3])
end

for i in 1:size(classTest)[1]
    vecsTest[i, :] = formulateText(m, test[i, 3])
end

train_mat = vecsTrain'
test_mat = vecsTest'

trainingdata = Flux.Data.DataLoader((train_mat, classTrain'), batchsize = 100, shuffle = true)
testingdata = Flux.Data.DataLoader((test_mat, classTest'))

function neural_net()
    nn = Chain(
        Dense(15, 1, x->σ.(x))
    )
end

neuralnet = neural_net()
opt = Descent(0.05)

lozz(x, y) = sum(Flux.Losses.binarycrossentropy(neuralnet(x), y))

para = Flux.params(neuralnet)

epochs = 500

for i in 1:epochs
    println(i)
    Flux.train!(lozz, para, trainingdata, opt)
end


#ROC Curve

preds = zeros(0)
for (x,y) in testingdata
   push!(preds, neuralnet(x)[1])
end

rocnums = MLBase.roc(classTest[:,end] .== 1,preds, 200)

TP = []
FP = []
for i in 1:length(rocnums)
    push!(TP,rocnums[i].tp/rocnums[i].p)
    push!(FP,rocnums[i].fp/rocnums[i].n)
end

Plots.plot!(FP,TP)
Plots.plot!((0:100)./100, (0:100)./100, leg = false)


#------Functions-----#
# Cleans up data a bit more before train/test split, samples data 50/50
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

# Formulates the text for the creation of the embeddings matrix
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

function createCorpusText(data,pads)
   allDocs = ""
   pad = ""
   for i in 1:3000
      pad = pad * " randomWordNow"
   end
   for i in 1:length(data[:, 3])
      println(i)
      if i != 1
         if data[i, 1] != data[i-1, 1] && i != 1
            print("This is a seperation")
            allDocs = allDocs * pad
         else
            allDocs = allDocs * " " * data[i, 3]
         end
      end
   end
   open("corpus.txt","a") do io
      println(io,allDocs)
   end
end
