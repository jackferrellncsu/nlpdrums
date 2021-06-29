using DataFrames
using Random
using Flux
using Word2Vec
using LinearAlgebra
using Languages
using BSON

function findNearest(model, vec, n)
    close = []
    closewords = []
    for i in 1:length(Word2Vec.vocabulary(model))
        w = Word2Vec.get_vector(model, Word2Vec.vocabulary(model)[i])
        push!(close, sum(w .* vec) / (norm(w)*norm(vec)))
    end
    for i in 1:n
        ind = argmax(close)
        push!(closewords, Word2Vec.vocabulary(model)[i])
        close[ind] = -1
    end
    return closewords
end

corp = open("/Users/mlovig/Downloads/text8") do file
    read(file, String)
end

Word2Vec.word2vec("/Users/mlovig/Downloads/text8", "text8Vectors.txt", size = 50, min_count = 0)
M = Word2Vec.wordvectors("text8Vectors.txt")

splittedWord = split(corp, " ")[2:end]
CurrWords = []
NextWords = []
#Forumulting scripts into vector form
countGoods = 0
countBads = 0
for i in 1:length(splittedWord)-1
    println(i/length(splittedWord))
    if string(splittedWord[i]) in Word2Vec.vocabulary(M)
        specWord = string(splittedWord[i])
        nextWord = string(splittedWord[i+1])
        if specWord ∉ stopwords(Languages.English())
            push!(CurrWords,
            convert(Vector{Float32},Word2Vec.get_vector(M,specWord)))
            push!(NextWords,
            convert(Vector{Float32},Word2Vec.get_vector(M,nextWord)))
            countGoods += 1
        elseif rand() < 1
            push!(CurrWords,
            convert(Vector{Float32},Word2Vec.get_vector(M,specWord)))
            push!(NextWords,
            convert(Vector{Float32},Word2Vec.get_vector(M,nextWord)))
            countBads += 1
        end
    end
end


#Training Nueral Net
nn = Chain(Flux.GRU(50,50),Dense(50,100,swish),Dense(100,150,swish),Dense(150,100,swish),Dense(100,50,swish),Dense(50,50, x->x))
opt = RADAM()
ps = Flux.params(nn)

function loss(x, y)
      return norm(nn(x)-y)
end


for i in 1:2000
    println(i)
    start = (i*2000)%(length(CurrWords)-2000)
    Flux.reset!(nn)
    Flux.train!(loss, ps, zip(CurrWords[start:start + 2000], NextWords[start:start + 2000]), opt)
end

using BSON: @save

@save "mymodel.bson" nn



#Calcuating nonconformity scores using NearestNeighbor
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
    push!(nonconf, ModelEstimateMeasure(nn,x,y))
end

#Grabbing p-values for our classification
epsilon = .05
pvals = []
for i in 1:length(testing[:,1])
    println(i)
    v = formulateText(M,testing[i,1])
    aTrue = ModelEstimateMeasure(nn,v,1)
    aFalse = ModelEstimateMeasure(nn,v,0)
    subpvals = []
    pTrue = sum(nonconf .> aTrue)/(length(nonconf)+1)
    pFalse = sum(nonconf .> aFalse)/(length(nonconf)+1)
    push!(subpvals,pFalse)
    push!(subpvals,pTrue)
    push!(pvals,subpvals)
end

#Creating Confomral Prediction intervals based on epsilon
preds = []
for i in 1:length(testing[:,1])
    println(i)
    subpreds = []
    if pvals[i][1] > epsilon
        push!(subpreds, 0)
    end
    if pvals[i][2] > epsilon
        push!(subpreds, 1)
    end
    push!(preds,subpreds)
end

#Calculating empirical condifence level
counter = 0
lengthI = 0
for i in 1:length(testing[:,2])
    if testing[i,2] in preds[i]
        counter += 1
    end
    lengthI += length(preds[i])
end

println(counter / length(testing[:,2]))
println(lengthI/length(testing[:,2]))


---


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
        if y == ys[i] && x != xs[i]
            push!(sameclass, norm(xs[i] .- x))
        elseif x != xs[i]
            push!(diffclass, norm(xs[i] .- x))
        end
    end
    return (minimum(sameclass)/minimum(diffclass))
end

function ModelEstimateMeasure(nn,x,y)
    if y == 1
        return nn(x)[1]
    else
        return 1 - nn(x)[1]
    end
end
