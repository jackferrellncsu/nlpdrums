using SparseArrays
    using JLD
    using Word2Vec
    using LinearAlgebra
    using Statistics
    using Embeddings
    using Flux
    using DataFrames
    using Lathe.preprocess: TrainTestSplit
    using Random
    using ProgressBars
    using StatsBase
    using Plots
    using ProgressBars
"""
    makeCorpus(filename<String>)

take in a txt filepath and outputs a cleaned string
"""
function makeCorpus(filename)
    f = open(filename, "r")
        corp = read(f, String)
        close(f)
        corp = replace(corp, "\r" => "")
        corp =replace(corp, "\n" => "")
        for i in 1:100
            corp = replace(corp, "chapter "*string(i) => "")
        end
        corp =replace(corp, "”" => "")
        corp =replace(corp, "“" => "")
        corp =replace(corp, ";" => "")
        corp = lowercase(corp)
        corp =replace(corp, "mr." => "mr")
        corp =replace(corp, "mrs." => "mrs")
        corp =replace(corp, "dr." => "dr")
        corp =replace(corp, "." => " . ")
        corp =replace(corp, "!" => " .")
        corp =replace(corp, "?" => " .")
        corp =replace(corp, ")" => "")
        corp =replace(corp, "(" => "")
        corp =replace(corp, "," => "")
        corp =replace(corp, "_" => "")
        corp =replace(corp, "—" => " ")
        corp =replace(corp, "-" => " ")
        corp =replace(corp, "—" => " ")
        corp =replace(corp, ". . ." => " ")
        corp =replace(corp, "*" => " ")
        corp =replace(corp, "’’" => " ")
    return corp
end

"""
    splitCorpus(corp<String>, minSize)

take in a corpus string and minSize sentance length and outputs array of sentances

Ex.
julia > splitCorpus("hello my name is. the dog barks alot")
    [["hello", "my", "name", "is"],["the", "dog", "barks", "alot"]]
"""
function splitCorpus(corp, minSize)
    vectorizedSpl = []
    spl = convert(Vector{String},split(corp, "."))
    for i in ProgressBar(1:length(spl))
        sent = convert(Vector{String},split(spl[i], " "))
        push!(vectorizedSpl, append!(sent, [String(".")]))
        vectorizedSpl[i] = filter(x->x≠"",vectorizedSpl[i])
    end
    return vectorizedSpl
end
"""
    permuteSentances(vectorizedSpl<Vector{String}>)

takes in an array of sentances and outputs every permutation based on order and the correspoding next word

Ex.
julia > permuteSentances([["hello", "my", "name"]])
    [[["hello"], ["hello", "my"]], ["my", "name"]]
"""
function permuteSentances(vectorizedSpl, minLength)
    sentances = []
    nextword = []
    for i in ProgressBar(1:length(vectorizedSpl))
        for ii in minLength:length(vectorizedSpl[i])-1
            push!(sentances,vectorizedSpl[i][1:ii])
            push!(nextword,vectorizedSpl[i][ii+1])
        end
    end
    return [sentances, nextword]
end

"""
    permuteSentances(sentances, nextword, get_word_index, incorrects)

takes in output from permuteSentances, a dictionary of embeddings an a ratio of incorrect examples to create the data set

"""
function makeDataRNN(sentances, nextword, get_word_index, D)

    sentancesVecs = []
    nextwordVecs = []
    for i in ProgressBar(1:length(sentances))
        temp = []
        if get(get_word_index,nextword[i],zeros(300)) != zeros(300)
            for x in S[i]
                if get(get_word_index,x,zeros(300)) != zeros(300)
                    push!(temp, get_word_index[x])
                end
            end
            if length(temp) > 0
                push!(sentancesVecs, temp)
                onehot = zeros(length(D))
                onehot[D[nextword[i]]] = 1
                push!(nextwordVecs, onehot)
            end
        end
    end

    return [sentancesVecs,nextwordVecs]
end

function toEmbedding(words, Embeddings)
    default = zeros(length(Embeddings["the"]))
    weight = 1/2^(length(words)-1)
    V = weight .* get(Embeddings,words[1],default)
    for x in words[2:end]
        V = V .+ weight .* get(Embeddings,x,default)
        weight *= 2
    end

    return convert(Vector{Float32},V)
end

function vecvec_to_matrix(vecvec,resp)
    dim1 = length(vecvec)
    dim2 = length(vecvec[1])+1
    my_array = zeros(Float32, dim1, dim2)
    for i in ProgressBar(1:dim1)
        for j in 1:dim2
            if j != dim2
                my_array[i,j] = vecvec[i][j]
            else
                my_array[i,end] = resp[i]
            end
        end
    end
    return my_array
end

function minNormD(P, C)
    minN = 100000
    for p in P
        normz = norm(p - C)
        if normz < minN
            minN = normz
        end
    end
    return minN
end

function sampleVecVec(mat,resp,spliter)
    partitionX1 = []
    partitionX2 = []
    partitionY1 = []
    partitionY2 = []
    inds = sample(1:length(mat), length(mat)-Int(ceil(spliter*length(mat))), replace = false)
    for i in ProgressBar(1:length(mat))
        if i in inds
            push!(partitionX2, mat[i])
            push!(partitionY2, resp[i])
        else
            push!(partitionX1, mat[i])
            push!(partitionY1, resp[i])
        end
    end
    return [partitionX1,partitionY1,partitionX2,partitionY2]
end
#------------------------
Corp = makeCorpus("/Users/mlovig/Downloads/1342-0.txt")

vectorizedSpl = splitCorpus(Corp,10)

#vectorizedSpl = vectorizedSpl[1:end-219]

wordspl = convert(Vector{String},split(Corp, " "))
wordspl = filter(x->(x≠"" && x≠"."),wordspl)
#wordspl = wordspl[1:findall(x -> x == ".***", wordspl)[1]-1]

uni = convert(Vector{String},unique(wordspl))
D = Dict(uni .=> 1:length(uni))

embtable = load_embeddings(GloVe{:en},4, keep_words=Set(uni))

get_word_index = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab))
get_vector_word = Dict(embtable.embeddings[:,ii]=>word for (ii,word) in enumerate(embtable.vocab))

#JLD.save("PridePrej.jld", "corpus", corp, "sentances", vectorizedSpl, "data", hcat(sentances,nextword), "embeddings", get_word_index)

S,N = permuteSentances(vectorizedSpl,5)

mat,resp = makeDataRNN(S,N, get_word_index, D)

trainingX, trainingY, extraX, extraY = sampleVecVec(mat,resp,.9)

calibX, calibY, testingX, testingY = sampleVecVec(extraX,extraY,.9)

rnn = Chain(
    LSTM(300, 300),
    LSTM(300, 3000),
    Dense(3000,  6958, x->x), softmax)
    ps = Flux.params(rnn)

function loss(x, y)
    Loss = 0
    for i in ProgressBar(1:length(x))
        Flux.reset!(rnn)
        yhat = rnn.(x[i])[end]
        Loss += Flux.Losses.crossentropy(yhat,y[i])
    end
    return Loss
end
function losssig(x, y)
    Flux.reset!(rnn)
    return Flux.Losses.binarycrossentropy(rnn.(x)[end],y)
end
losses = []
batch  = 100
epochs = 1
eta = .001
DL = Flux.Data.DataLoader((trainingX,trainingY), batchsize = 1000, shuffle = true)
for i in ProgressBar(1:epochs)
    opt = Descent(eta)
    Flux.train!(loss, ps, DL, opt)
    L = loss(trainingX,trainingY)
    if i > 1 && L > losses[end]
        eta = eta*.9
    end
    push!(losses,L)
    Flux.reset!(rnn)
    print("         ", L, "        ", sum(rnn.(trainingX[i])[end] .* trainingY[i]),"       ")
end

for i in 1:epochs
    inds = shuffle(1:length(trainingX))
    trainingX = trainingX[inds, :]
    trainingY = trainingY[inds, :]
    for ii in ProgressBar(1:Int(floor(length(trainingX)/batch)))
        if ii%10 == 9
            opt = RADAM(eta)
            eta = eta*.8
        end
        L = sum(loss(trainingX[(ii-1) * batch + 1:ii*batch], trainingY[(ii-1) * batch + 1:ii*batch]))
        Flux.train!(losssig, ps, zip(trainingX[(ii-1) * batch + 1:ii*batch], trainingY[(ii-1) * batch + 1:ii*batch]), opt)
        push!(losses, L/batch)

        print("         ", L/batch)
    end
end



#---------------------
occur = countmap(wordspl)
occurvec = []
for x in uni
    push!(occurvec, occur[x])
end
a_i = zeros(length(calibX))
    for i in ProgressBar(1:size(calibX)[1])
        Flux.reset!(rnn)
        a_i[i] = (1 - sum(rnn.(calibX[i])[end] .* calibY[i]))
    end

correct = []
    eff = []
    epsilon = .05
    Q = quantile(a_i[1:692],1-epsilon)
    for ii in ProgressBar(1:length(testingX))
        Pred = (1 .- rnn.(testingX[ii])[end]) .< Q
        push!(correct, Flux.onecold(testingY[ii], uni) in uni[Pred])
        push!(eff, sum(Pred))
        print("          ", 1-mean(correct), "         ", median(eff), "         ",quantile(eff, .9) - quantile(eff, .1), "                                         ")
    end


A = reverse(sort(collect(zip(values(occur),keys(occur)))))
a = []
for i in 1:length(A)
    push!(a,A[i][1])
end
a = a / sum(a)
suma = []
for i in 1:length(a)
    push!(suma,sum(a[1:i]))
end
diffa = suma .> .95
argmax(diffa)

norms = []
for i in ProgressBar(1:length(uni))
    for ii in uni
        push!(norms,norm(toEmbedding([uni[i]],get_word_index)-toEmbedding([ii],get_word_index)))
    end
end

using BSON: @save

BSON.@save "BestModelLarge.bson" nn

using BSON: @load

BSON.@load "/Users/mlovig/Documents/GitHub/nlpdrums/BestModel.bson" nn
