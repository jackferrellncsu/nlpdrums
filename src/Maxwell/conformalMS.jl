using DataFrames
using Random
using Flux
using Word2Vec
using LinearAlgebra
using NNlib
using Lathe.preprocess: TrainTestSplit

text = CSV.read("/Users/mlovig/Downloads/archive/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv", DataFrame)[:,21]
scores = (CSV.read("/Users/mlovig/Downloads/archive/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv", DataFrame)[:,19])
ratings = []
for i in 1:length(scores)
    push!(ratings, Flux.onehot(scores[i], 1:5))
end
training,testing = TrainTestSplit(DataFrame(hcat(text,ratings)),.8)
training = training[shuffle(axes(training, 1)), :]
corp = join(training[:,1])

#Creating corpus and training vectors
open("movieCorpus.txt", "w") do io
    write(io, corp)
end;

Word2Vec.word2vec("movieCorpus.txt", "movieVectors.txt", size = 300, min_count = 1)
M = Word2Vec.wordvectors("movieVectors.txt")

#Forumulting scripts into vector form
Dtrain = training[1:Int(ceil(size(training)[1]*.8)),:]
calib = training[Int(ceil(size(training)[1]*.8))+1:end,:]
for i in 1:length(Dtrain[:,1])
    println(i)
    Dtrain[i,1] = convert(Vector{Float32},formulateText(M,training[i,1]))
end
for i in 1:length(calib[:,1])
    println(i)
    calib[i,1] = convert(Vector{Float32},formulateText(M,calib[i,1]))
end

#Training Nueral Net
nn = Chain(
    Dense(300, 200, gelu),Dense(200, 100, gelu),
    Dense(100, 50, gelu),Dense(50, 5), softmax
    )
opt = RADAM()
ps = Flux.params(nn)

function loss(x, y)
  if rand() <= .1
     return sum(Flux.Losses.crossentropy(nn(x), y))
  else
     return 0
  end
end

for i in 1:100
    println(i)
    Flux.train!(loss, ps, zip(Dtrain[:,1], Dtrain[:,2]), opt)
end

#Calculating nonconformity scores using NearestNeighbor
#=
nonconf = []
for (x,y) in zip(calib[:,1],calib[:,2])
    push!(nonconf, NearestNeighbor(calib[:,1], calib[:,2], x,y))
end

#Grabbing p-values for our classification
epsilon = .05
pvals = []
for i in 1:length(testing[:,1])
    println(i)
    aTrue = NearestNeighbor(calib[:,1],calib[:,2],formulateText(M,testing[i,1]),1)
    aFalse = NearestNeighbor(calib[:,1],calib[:,2],formulateText(M,testing[i,1]),0)
    subpvals = []
    pTrue = sum(nonconf .> aTrue)/(length(nonconf)+1)
    pFalse = sum(nonconf .> aFalse)/(length(nonconf)+1)
    push!(subpvals,pFalse)
    push!(subpvals,pTrue)
    push!(pvals,subpvals)
end
=#


#Nonconformity using model estimate
nonconf = []
for (x,y) in zip(calib[:,1],calib[:,2])
    #push!(nonconf, ModelEstimateMeasure(nn,x,argmax(y)))
    push!(nonconf, NearestNeighbor(calib[:,1],calib[:,2],x,y))
end

#Grabbing p-values for our classification
pvals = []
for i in 1:length(testing[:,1])
    println(i/length(testing[:,1]))
    v = formulateText(M,testing[i,1])
    subpvals = []
    for ii in 1:5
        #acase = ModelEstimateMeasure(nn,v,ii)
        acase = NearestNeighbor(calib[:,1],calib[:,2],v,ii)
        pcase = sum(nonconf .> acase)/(length(nonconf)+1)
        push!(subpvals,acase)
    end
    push!(pvals,subpvals)
end

#Creating Confomral Prediction intervals based on epsilon
epsilon = .1
    preds = []
    for i in 1:length(testing[:,1])
    subpreds = []
    for ii in 1:5
        if pvals[i][ii]>epsilon
            push!(subpreds, ii)
        end
    end
    push!(preds,subpreds)
end

#Calculating empirical condifence level
counter = 0
lengthI = 0
for i in 1:length(testing[:,2])
    if argmax(testing[i,2]) in preds[i]
        counter += 1
    end
    lengthI += length(preds[i])
end

println(counter / length(testing[:,2]))
println(lengthI/length(testing[:,2]))


---

function makeCSVMulti()
    trainingScores = []
    trainingReviews = []
    for i in 1:12500
        println(i)
        for ii in 6:10
            if isfile("/Users/mlovig/Downloads/aclImdb/train/pos/$i"*"_"*"$ii.txt")
            f = open("/Users/mlovig/Downloads/aclImdb/train/pos/$i"*"_"*"$ii.txt", "r")
            push!(trainingReviews,read(f, String))
            close(f)
            push!(trainingScores, Flux.onehot(ii, 1:10))
            end
        end
    end
    for i in 1:12500
        println(i)
        for ii in 0:6
            if isfile("/Users/mlovig/Downloads/aclImdb/train/neg/$i"*"_"*"$ii.txt")
            f = open("/Users/mlovig/Downloads/aclImdb/train/neg/$i"*"_"*"$ii.txt", "r")
            push!(trainingReviews,read(f, String))
            close(f)
            push!(trainingScores, Flux.onehot(ii, 1:10))
            end
        end
    end
    dfTrain = DataFrame(hcat(trainingReviews,trainingScores))

    testScores = []
    testReviews = []

    for i in 1:12500
        println(i)
        for ii in 6:10
            if isfile("/Users/mlovig/Downloads/aclImdb/test/pos/$i"*"_"*"$ii.txt")
            f = open("/Users/mlovig/Downloads/aclImdb/test/pos/$i"*"_"*"$ii.txt", "r")
            push!(testReviews,read(f, String))
            close(f)
            push!(testScores, Flux.onehot(ii, 1:10))
            end
        end
    end
    for i in 1:12500
        println(i)
        for ii in 0:6
            if isfile("/Users/mlovig/Downloads/aclImdb/test/neg/$i"*"_"*"$ii.txt")
            f = open("/Users/mlovig/Downloads/aclImdb/test/neg/$i"*"_"*"$ii.txt", "r")
            push!(testReviews,read(f, String))
            close(f)
            push!(testScores, Flux.onehot(ii, 1:10))
            end
        end
    end

    dfTest = DataFrame(hcat(testReviews,testScores))

    return [dfTrain,dfTest]
end

function MakeCSV()
    trainingScores = []
    trainingReviews = []

    for i in 1:12500
        println(i)
        for ii in 6:10
            if isfile("/Users/mlovig/Downloads/aclImdb/train/pos/$i"*"_"*"$ii.txt")
            f = open("/Users/mlovig/Downloads/aclImdb/train/pos/$i"*"_"*"$ii.txt", "r")
            push!(trainingReviews,read(f, String))
            close(f)
            push!(trainingScores, 1)
            end
        end
    end
    for i in 1:12500
        println(i)
        for ii in 0:6
            if isfile("/Users/mlovig/Downloads/aclImdb/train/neg/$i"*"_"*"$ii.txt")
            f = open("/Users/mlovig/Downloads/aclImdb/train/neg/$i"*"_"*"$ii.txt", "r")
            push!(trainingReviews,read(f, String))
            close(f)
            push!(trainingScores, 0)
            end
        end
    end
    dfTrain = DataFrame(hcat(trainingReviews,trainingScores))

    testScores = []
    testReviews = []

    for i in 1:12500
        println(i)
        for ii in 6:10
            if isfile("/Users/mlovig/Downloads/aclImdb/test/pos/$i"*"_"*"$ii.txt")
            f = open("/Users/mlovig/Downloads/aclImdb/test/pos/$i"*"_"*"$ii.txt", "r")
            push!(testReviews,read(f, String))
            close(f)
            push!(testScores, 1)
            end
        end
    end
    for i in 1:12500
        println(i)
        for ii in 0:6
            if isfile("/Users/mlovig/Downloads/aclImdb/test/neg/$i"*"_"*"$ii.txt")
            f = open("/Users/mlovig/Downloads/aclImdb/test/neg/$i"*"_"*"$ii.txt", "r")
            push!(testReviews,read(f, String))
            close(f)
            push!(testScores, 0)
            end
        end
    end

    dfTest = DataFrame(hcat(testReviews,testScores))

    return [dfTrain,dfTest]

end

function formulateText(model, script)
   words = split(script, " ")
   vecs = zeros(length(Word2Vec.get_vector(model,Word2Vec.vocabulary(model)[1])))
   for i in words[1:end]
      if i in Word2Vec.vocabulary(model) #&& i ∉ stopwords(Languages.English())
         vecs = vecs .+ Word2Vec.get_vector(model,i)
      end
   end
   return vecs
end

function NearestNeighbor(xs,ys,x,y)
    sameclass = []
    diffclass = []
    for i in 1:length(ys)
        if y == argmax(ys[i]) && x != xs[i]
            push!(sameclass, norm(xs[i] .- x))
        elseif x != xs[i]
            push!(diffclass, norm(xs[i] .- x))
        end
    end
    return (minimum(sameclass)/minimum(diffclass))
end

function ModelEstimateMeasure(nn,x,y)
    return nn(x)[y]
end
