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
function makeData(sentances, nextword, get_word_index, incorrects)

    sentancesVecs = []
    nextwordVecs = []
    for i in ProgressBar(1:length(sentances))
        if isnan(toEmbedding(sentances[i],get_word_index)[1]) != 1 &&
            isnan(toEmbedding([nextword[i]],get_word_index)[1]) != 1&&
            toEmbedding([nextword[i]],get_word_index)[1] != 0 &&
            toEmbedding(sentances[i],get_word_index)[1] != 0

                push!(sentancesVecs, toEmbedding(sentances[i],get_word_index))
                push!(nextwordVecs, toEmbedding([nextword[i]],get_word_index))
        end
    end

    avoid = Dict()
    for i in ProgressBar(1:length(sentancesVecs))
        avoid[sentancesVecs[i]] = append!(get(avoid, sentancesVecs[i], []), [nextwordVecs[i]])
    end

    temp = vcat.(sentancesVecs,nextwordVecs)
    sentancesVecs = temp

    bogusNextWord = []
    counter = 0
    minnormmD = []
    while length(bogusNextWord) < incorrects*length(sentancesVecs)
        if length(bogusNextWord) % length(sentancesVecs) == 0
            println(length(bogusNextWord) / (incorrects*length(sentancesVecs)))
        end
        W = get(get_word_index,rand(uni),zeros(300))
        Context = sentancesVecs[(counter % length(sentancesVecs))+1][1:300]
        if  isnan(W[1]) != 1 && W[1] != 0 && minNormD(avoid[Context], W) > 7.75
            counter += 1
            push!(bogusNextWord, vcat(Context, W))
        end
    end

    len = length(sentancesVecs)

    println("Made Data")

    append!(sentancesVecs,bogusNextWord)

    println("Concatanated Data")

    resp = Int.(vcat(ones(Int(length(sentancesVecs)/(incorrects+1))),zeros(Int(incorrects*length(sentancesVecs)/(incorrects+1)))))
    mat = vecvec_to_matrix(sentancesVecs,resp)

    return [mat,resp]
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

#------------------------
Corp = makeCorpus("/Users/mlovig/Downloads/1342-0.txt")

vectorizedSpl = splitCorpus(Corp,10)

#vectorizedSpl = vectorizedSpl[1:end-219]

wordspl = convert(Vector{String},split(Corp, " "))
wordspl = filter(x->(x≠"" && x≠"."),wordspl)
#wordspl = wordspl[1:findall(x -> x == ".***", wordspl)[1]-1]

uni = convert(Vector{String},unique(wordspl))
D = Dict(1:length(uni) .=> uni)

embtable = load_embeddings(GloVe{:en},4, keep_words=Set(uni))

get_word_index = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab))
get_vector_word = Dict(embtable.embeddings[:,ii]=>word for (ii,word) in enumerate(embtable.vocab))

#JLD.save("PridePrej.jld", "corpus", corp, "sentances", vectorizedSpl, "data", hcat(sentances,nextword), "embeddings", get_word_index)

S,N = permuteSentances(vectorizedSpl,5)

mat,resp = makeData(S,N, get_word_index, 24)
mat = matNN
df = DataFrame(mat)

training,extra = TrainTestSplit(df, .9)
calib,testing = TrainTestSplit(extra, .9)

training = Matrix(training)
testing= Matrix(testing)
calib = Matrix(calib)

nn = Chain(
    Dense(600, 300, gelu),
    Dense(300, 100, gelu),
    Dense(100, 75, gelu),
    Dense(50,  25,  gelu),
    Dense(25,  10,  gelu),
    Dense(10,  1, sigmoid))
    ps = Flux.params(nn)

function loss(x, y)
    return Flux.Losses.binarycrossentropy(nn(x),y)
end

losses = []
batch  = 5000
epochs = 50
eta = .00001

DL = Flux.Data.DataLoader((training[:,1:end-1]',training[:,end]'), batchsize = 50000, shuffle = true)
opt = RADAM(eta)
for i in ProgressBar(1:epochs)
    Flux.train!(loss, ps, DL, opt)
end

for i in 1:epochs
    training = training[shuffle(1:end), :]
    vecsvecs = []
    for i in ProgressBar(1:size(training)[1])
        push!(vecsvecs, training[i, 1:end-1])
    end
    for ii in ProgressBar(1:Int(floor(length(vecsvecs)/batch)))
        if ii%10 == 9
            opt = RADAM(eta)
            eta = eta*.8
        end
        L = sum(loss.(vecsvecs[(ii-1) * batch + 1:ii*batch], training[(ii-1) * batch + 1:ii*batch,end]))
        Flux.train!(loss, ps, zip(vecsvecs[(ii-1) * batch + 1:ii*batch], training[(ii-1) * batch + 1:ii*batch,end]), opt)
        push!(losses, L/batch)

        print("         ", L/batch)
    end
end

#---------------------
occur = countmap(wordspl)
a_i = []
    for i in 1:size(calib)[1]
    if isnan(calib[i,1]) == false && (calib[i,end]) == 1 && calib[i,end-1] != 0
        #LL = log(occur[get_vector_word[calib[i,301:600]]]) + .000000001
        #LLA = 1 - nn(calib[i,1:end-1])[1]^(1/LL)
        NNA = 1 - nn(calib[i,1:end-1])[1]
        #push!(a_i,  max(NNA,LLA))
        push!(a_i,  NNA)
    end
end
epsilon = .1
ii=1
Q = quantile(a_i,1-epsilon)
as = []
if testing[ii,301:600] != zeros(300) && testing[ii,end] == 1
    pred = []

    for i in 1:length(uni)
        if get(get_word_index, uni[i], 0)!= 0
            LL = log2(occur[uni[i]]) + .000000001
            LLA = (1 - nn(vcat(testing[ii,1:300], toEmbedding([uni[i]],get_word_index)))[1]^(1/LL))
            NNA = 1 - nn(vcat(testing[ii,301:600], toEmbedding([uni[i]],get_word_index)))[1]
            #a = max(NNA,LLA)
            a = LLA
            push!(as,a)
            if a < Q
                push!(pred, uni[i])
            end
        end
    end
    correctword = get_vector_word[testing[ii,301:600]]
    println(ii, " ", length(pred), " ", correctword in pred, " ", correctword)
    push!(counter, correctword in pred)
    push!(eff, length(pred))
    println(ii / sum(testing[:,end]), " ", sum(counter)/length(counter))
end


correct = []
    eff = []
    epsilon = .1
    Q = quantile(a_i,1-epsilon)
    for ii in ProgressBar(1:argmin(testing[:,end]))
        if testing[ii,end] == 1 && testing[ii,600] != 0
            pred = []
            for i in 1:length(uni)
                if get(get_word_index, uni[i], 0)!= 0
                    #LL = ln(occur[uni[i]]) + .000000001
                    #LLA = (1 - nn(vcat(testing[ii,1:300], toEmbedding([uni[i]],get_word_index)))[1]^(1/LL))
                    NNA = 1 - nn(vcat(testing[ii,301:600], toEmbedding([uni[i]],get_word_index)))[1]
                    #a = max(NNA,LLA)
                    a = NNA
                    if a <= Q
                        push!(pred, uni[i])
                    end
                end
            end
            correctword = get_vector_word[testing[ii,301:600]]
            #println(ii, " ", length(pred), " ", correctword in pred, " ", correctword)
            push!(correct, correctword in pred)
            push!(eff, length(pred))
            print("          ", 1-mean(correct), "         ", median(eff), "         ",quantile(eff, .9) - quantile(eff, .1))
        end
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
