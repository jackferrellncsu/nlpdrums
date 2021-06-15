using Flux
using Lathe
using MLBase
using Plots

include("../data_cleaning.jl")
include("../embeddings_nn.jl")

#Importing data and creating corpus
data = importClean()
sort!(data, "medical_specialty")
rm("corpus.txt")
createCorpusText(data, 0)

#Create embeddings
vecfile = @__DIR__
vecfile = vecfile * "/vectors.txt"

veclength = 15
word2vec("corpus.txt", vecfile; size = veclength, window = 15, min_count = 5)

#Extract embeddings
embeddings = wordvectors(vecfile)


field = " Cardiovascular / Pulmonary"
#Balance field of interest with negative samples
sample = filtration(data, field)
#Split into training and testing
train, test =  TrainTestSplit(sample, 0.9)

yTrain = train[:, 1] .== field
yTrain *= 1.0
yTest = test[:, 1] .== field
yTest *= 1.0


#Put embeddings into matrix form

vecsTrain = zeros(length(yTrain), veclength)
vecsTest = zeros(length(yTest), veclength)

for i in 1:length(yTrain)
    vecsTrain[i, :] = formulateText(embeddings, train[i, 3])
end

for i in 1:length(yTest)
    vecsTest[i, :] = formulateText(embeddings, test[i, 3])
end

train_mat = vecsTrain'
test_mat = vecsTest'


#Make matrices into flux format
trainingdata = Flux.Data.DataLoader((train_mat, yTrain'), batchsize = 100, shuffle = true)
testingdata = Flux.Data.DataLoader((test_mat, yTest'))

#Train first model with hardsigmoid
model1 = Chain(
            Dense(15, 7, hardsigmoid),
            Dense(7, 1, x -> σ.(x))
        )

opt = Descent(0.05)

loss(x, y) = sum(Flux.Losses.binarycrossentropy(model1(x), y))
para = Flux.params(model1)

epoch = 1000

for i in 1:epoch
    Flux.train!(loss, para, trainingdata, opt)
end

Flux.params(model1)

acc = 0
pred = Vector{Float64}()
for (x, y) in testingdata
    #Note: although broadcasting is being used here,
    #these are not vectors, they are matrices with single values
    global acc += sum((model1(x) .> .5) .== y)
    push!(pred, model1(x)[1, 1])
end
print(1- acc / length(yTest))

#Save model
using BSON: @save

@save "src/Jack/jackModel.bson" model1

r = MLBase.roc(convert(BitVector, yTest), pred, 100)
fpr = false_positive_rate.(r)
tpr = true_positive_rate.(r)

plot(fpr, tpr)

function sumEmbeddings(embeddings, transcript)
    words = split(transcript, " ")
    scriptvec = zeros(length(get_vector(embeddings, "the")))
    for word in words
        if word ∉ stopwords(Languages.English()) && word ∈ embeddings.vocab
            scriptvec += get_vector(embeddings, word)
        end
    end
    return scriptvec
end
