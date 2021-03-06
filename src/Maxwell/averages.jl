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
            isnan(toEmbedding([nextword[i]],get_word_index)[1]) != 1#&&
            zeros(300) != toEmbedding([nextword[i]],get_word_index) &&
            zeros(300) != toEmbedding(sentances[i],get_word_index)

                push!(sentancesVecs, toEmbedding(sentances[i],get_word_index))
                push!(nextwordVecs, toEmbedding([nextword[i]],get_word_index))
        end
    end

    temp = vcat.(sentancesVecs,nextwordVecs)
    sentancesVecs = temp

    bogusNextWord = []
    counter = 0
    while length(bogusNextWord) < incorrects*length(sentancesVecs)
        if length(bogusNextWord) % length(sentancesVecs) == 0
            println(length(bogusNextWord) / (incorrects*length(sentancesVecs)))
        end

        W = get(get_word_index,rand(uni),zeros(300))
        if  isnan(W[1]) != 1 && zeros(300) != W && norm(W-sentancesVecs[(counter % length(sentancesVecs))+1][301:600]) > 6
            Context = sentancesVecs[(counter % length(sentancesVecs))+1][1:300]
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
    return mat,resp
end

function toEmbedding(words, Embeddings)
    default = zeros(length(Embeddings["the"]))
    #weight = 1/2^(length(words)-1)
    weight = 1/2
    #V = weight .* get(Embeddings,words[1],default)
    V = 1 .* get(Embeddings,words[end],default)
    #for (i,x) in zip(1:length(words[2:end]), words[2:end])
    for (i,x) in zip(1:length(words[2:end]), reverse(words[2:end-1]))
        if i >= 3
            V = V .+ weight .* get(Embeddings,x,default)
            weight /= 2
            #weight *= 2
        end
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

S,N = permuteSentances(vectorizedSpl,1)
SS = vcat.(S,N)
SSC = countmap(SS)
SSCA = [v for (k,v) in SSC]
mat,resp = makeData(S,N, get_word_index, 0)
mat = mat[:,1:end-1]

df = DataFrame(mat)

training,extra = TrainTestSplit(df, .9)
calib,testing = TrainTestSplit(extra, .9)

training = Matrix(training)
testing= Matrix(testing)
calib = Matrix(calib)

wordContextVectors = Dict()
    for i in ProgressBar(1:size(training)[1])
        if training[i,end] != 0
            word = get_vector_word[training[i,301:end]]
            wordContextVectors[word] = append!(get(wordContextVectors, word, []), [training[i,1:300]])
        end
    end

wordOccurance = Dict()
    for x in uni
        wordOccurance[x] = length(get(wordContextVectors,x,[]))
    end

a_i = []
    for i in ProgressBar(1:size(calib)[1])
    if calib[i,end] != 0
        word = get_vector_word[calib[i,301:end]]
        P = get(wordContextVectors, word, 0)
        if  P != 0
            X = wordOccurance[word]+4
            T = log(X)
            #push!(a_i, norm(mean(P) - calib[i,1:300]))
            #push!(a_i, minNorm(P,calib[i,1:300]))
            push!(a_i, minNorm(P,calib[i,1:300])/T)
        end
    end
    end
PPP = []
correct = []
    eff = []
    epsilon = .05
    Q = quantile(a_i, 1-epsilon)
    for i in ProgressBar(1:size(testing)[1])
    if testing[i,end] != 0
        pred = []
        for ii in 1:length(uni)
            P = get(wordContextVectors, uni[ii], 0)
            if  P != 0
                X = wordOccurance[uni[ii]] + 4
                T = log(X)
                #if norm(mean(P) - testing[i,1:300]) <= Q
                #if minNorm(P,testing[i,1:300]) <= Q
                if minNorm(P,testing[i,1:300])/T <= Q
                    push!(pred, uni[ii])
                end
            end
        end
        trueWord = get_vector_word[testing[i,301:600]]
        push!(correct,trueWord in pred)
        push!(eff, length(pred))
        PPP = pred
        print("         ",1-mean(correct),"         ", mean(eff), "         ", median(eff), "         " ,quantile(eff,.75)-quantile(eff,.25))
    end
    end

function minNorm(P, C)
    normz = []
    for p in P
        push!(normz, norm(p - C))
    end
    return minimum(normz)
end

function meanNorm(P, C)
    normz = []
    for p in P
        push!(normz, norm(p - C))
    end
    if length(normz) == 0
        return 9
    end
    return mean(normz)
end
