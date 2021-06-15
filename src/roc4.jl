using Flux
using Lathe
using MLBase
using Plots

include("data_cleaning.jl")
include("embeddings_nn.jl")

#=================Embeddings creation=================================#
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

testingdata = Flux.Data.DataLoader((test_mat, yTest'))
#==============================================================================#

#============================Jack ROC==========================================#
using BSON: @load

@load "src/Jack/jackModel.bson" model1

jackpreds = Vector{Float64}()
for (x, y) in testingdata
    push!(jackpreds, model1(x)[1, 1])
end

jackr = roc(convert(BitVector, yTest), jackpreds)

jacktpr = true_positive_rate.(jackr)
jackfpr = false_positive_rate.(jackr)

#==============================================================================#

#=============Compute your roc data here ====================#




#======use plot! to add your roc curves to plot==============#
plot(title = "ROC Comparisons")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")
plot!(jackfpr, jacktpr, label = "W2V NN")
