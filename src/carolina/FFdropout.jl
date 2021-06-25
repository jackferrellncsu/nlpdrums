using Lathe
using Flux
using Word2Vec
include("../embeddings_nn.jl")
include("../data_cleaning.jl")

#import text, sort by field, and create corpus
text = importClean()
train, test = TrainTestSplit(text, 0.9)
sort!(train, "medical_specialty")
createCorpusText(train, 0)

pwd()
field = " Cardiovascular / Pulmonary"

word2vec("corpus.txt", "vectors.txt", size = 15, window = 20)
rm("corpus.txt")

m = wordvectors("vectors.txt", normalize = true)

dataS = filtration(text, field)


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
        Dense(15, 7, mish),
        Dropout(0.6),
        Dense(7, 1, x->σ.(x))
    )
end

neuralnet = neural_net()
Flux.testmode!(neuralnet)
opt = RADAM()

lozz(x, y) = sum(Flux.Losses.binarycrossentropy(neuralnet(x), y))

para = Flux.params(neuralnet)

# training the model

epochs = 1000

for i in 1:epochs
    println(i)
    Flux.train!(lozz, para, trainingdata, opt)
end


# prints predictions vs actual
# and error rate
acc = 0
for (x, y) in testingdata
    print(neuralnet(x))
    global acc+=sum((neuralnet(x).>0.5) .== y)
    print(neuralnet(x) .> .5, " : ")
    println(y)
end
println(1 - acc/length(classTest))

#= Trace Plot ??
x = 1:epochs
y = traceY
Plots.plot(x, y, label = "Loss Progression")
xlabel!("Total # of epochs")
ylabel!("Loss values")
=#
