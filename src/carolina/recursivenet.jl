using Lathe
using Flux
include("../embeddings_nn.jl")
include("../data_cleaning.jl")

text = importClean()
sort!(text, "medical_specialty")
createCorpusText(text, 10)

field = " Cardiovascular / Pulmonary"
word2vec("corpus.txt", "vectors.txt", size = 15, window = 20)

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
        Dense(15, 7, hardσ),
        Dense(7, 1, x->σ.(x))
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


acc = 0
for (x, y) in testingdata
    acc+=sum((neuralnet(x).>0.5) .== y)
    print(neuralnet(x) .> .5, " : ")
    println(y)
end
println(1 - acc/length(classTest))
