using Flux
using Lathe
using MLBase
using Plots
using Random
using GLM

Random.seed!(1234)
include("data_cleaning.jl")
include("embeddings_nn.jl")
include("DTMCreation.jl")

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
dataS = filtration(data, field)
#Split into training and testing
train, test =  TrainTestSplit(data, 0.9)

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

jackr = MLBase.roc(convert(BitVector, yTest), jackpreds)

jacktpr = true_positive_rate.(jackr)
jackfpr = false_positive_rate.(jackr)

#==============================================================================#

#=============Compute your roc data here ====================#
#====================Emi ROC===================================================#
dtmi = CreateDTM(data, field)
df = DataFrame(1.0*dtmi', :auto)

pcanntrain, pcanntest = TrainTestSplit(df, .9)

numPC = 27
# PCA/SVD matrices
Us, sigs, Vts = PCAVecs(Matrix(pcanntrain)[:, 1:end - 1], 50)
U = Us[NumPC]
sig = sigs[NumPC]
Vt = Vts[NumPC]

UsTest, sigsTest, VtsTest = PCAVecs(Matrix(pcanntest)[:, 1:end - 1], 50)
UTest = UsTest[NumPC]
STest = sigsTest[NumPC]
VTest = VtsTest[NumPC]

pcaNNTrainY = pcanntrain[:,end]
pcaNNTestY = pcanntest[:,end]

# creating the matrix to run through nn
train_mat_pca_nn = U'
test_mat_pca_nn = UTest'

@load "src/emi/nnPCA.bson" nnPCA
pcaNNTestData = Flux.Data.DataLoader((test_mat_pca_nn, pcaNNTestY'))

pcaNNPreds = Vector{Float64}()
for (x, y) in pcaNNTestData
    push!(pcaNNPreds, nnPCA(x)[1, 1])
end

pcaNNRocs = MLBase.roc(convert(BitVector, pcaNNTestY), pcaNNPreds)

emiTPR = true_positive_rate.(pcaNNRocs)
emiFPR = false_positive_rate.(pcaNNRocs)

#========================Carolina ROC==========================================#
#(Logit with embeddings)
@load "src/carolina/logitEmbeddings.bson" neuralnet

w2vlogitpreds = Vector{Float64}()
for (x, y) in testingdata
    push!(w2vlogitpreds, neuralnet(x)[1, 1])
end

w2vLogitRoc = MLBase.roc(convert(BitVector, yTest), w2vlogitpreds,)

carolinaTPR = true_positive_rate.(w2vLogitRoc)
carolinaFPR = false_positive_rate.(w2vLogitRoc)

#===========================Max ROC============================================#
#PCA logit
pcalogittrain = pcanntrain
pcalogittest = pcanntest

ii = 25
Us, Sigs, Vts = PCAVecs(Matrix(pcalogittrain)[:, 1:end - 1], 25)
dftrain = DataFrame(hcat(Us[ii],pcalogittrain[:,end]), :auto)

z=term(Symbol(:x, ii+1)) ~ term(0) + sum(term.(Symbol.(names(dftrain[:, Not(Symbol(:x, ii+1))]))))
logit = glm(z,dftrain, Bernoulli(), LogitLink())
beta = Vts[ii]'*inv(diagm(Sigs[ii]))*coef(logit)
rets = Matrix(pcalogittest)[:,1:end - 1]*hcat(beta)
bin = 1 ./ (1 .+ exp.(-rets))
maxROC = MLBase.roc(pcalogittest[:,end] .== 1,vec(bin))

maxTPR = true_positive_rate.(maxROC)
maxFPR = false_positive_rate.(maxROC)
#======use plot! to add your roc curves to plot==============#
plot((0:100)./100, (0:100)./100, title = "ROC Comparisons", label = "50/50 guess")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")
plot!(jackfpr, jacktpr, label = "W2V NN")
plot!(emiFPR, emiTPR, label = "PCA NN")
plot!(carolinaFPR, carolinaTPR, label = "W2V Logit")
plot!(maxFPR, maxTPR, label = "PCA Logit")
