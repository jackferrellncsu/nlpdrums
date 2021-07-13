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
function permuteSentances(vectorizedSpl)
    sentances = []
    nextword = []
    for i in ProgressBar(1:length(vectorizedSpl))
        for ii in 1:length(vectorizedSpl[i])-1
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
    for i in ProgressBar(1:length(sentances))
        if NaN ∉ toEmbedding(sentances[i],get_word_index) &&
            NaN ∉ toEmbedding([nextword[i]],get_word_index) &&
            zeros(300) != toEmbedding([nextword[i]],get_word_index) &&
            zeros(300) != toEmbedding(sentances[i],get_word_index)
                push!(sentancesVecs, vcat(toEmbedding(sentances[i],get_word_index),toEmbedding([nextword[i]],get_word_index)))
        end
    end

    bogusNextWord = []
    counter = 0
    while length(bogusNextWord) < incorrects*length(sentancesVecs)
        if length(bogusNextWord) % length(sentancesVecs) == 0
            println(length(bogusNextWord) / (incorrects*length(sentancesVecs)))
        end
        W = get(get_word_index,rand(uni),zeros(300))
        if NaN ∉ W && zeros(300) != W
            Context = sentancesVecs[(counter % length(sentancesVecs))+1]
            counter += 1
            push!(bogusNextWord, vcat(Context, W))
        end
    end

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
                my_array[i,j] = resp[i]
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

S,N = permuteSentances(vectorizedSpl)
mat,resp = makeData(S,N, get_word_index, 25)

df = DataFrame(mat)

training,testing = TrainTestSplit(df, .9)
training,calib = TrainTestSplit(training, .9)

training = Matrix(training)
testing= Matrix(testing)
calib = Matrix(calib)

nn = Chain(
    Dense(600,300, gelu),
    Dense(300,150, gelu),
    Dense(150, 75, gelu),
    Dense( 75, 25, gelu),
    Dense( 25,  1, sigmoid)
    )

opt = RADAM(.00001)
ps = Flux.params(nn)

function loss(x, y)
    L = Flux.Losses.binarycrossentropy(nn(x),y)
    if isnan(L)
        return 0
    else
        return L
    end
end

losses = []
batch = 5000
epochs = 10
eta = .00001

DL = Flux.Data.DataLoader((training[:,1:end-1]',training[:,end]'), batchsize = batch, shuffle = true)

for i in ProgressBar(1:epochs)
    opt = RADAM(eta)
    eta = eta/2
    Flux.train!(loss, ps, DL, opt)
end

for i in 1:epochs
    opt = RADAM(eta)
    eta = eta / 2
    training = training[shuffle(1:end), :]
    vecsvecs = []
    for i in 1:size(training)[1]
        push!(vecsvecs, training[i, 1:end-1])
    end
    for ii in ProgressBar(1:Int(floor(length(vecsvecs)/batch)))
        Flux.train!(loss, ps, zip(vecsvecs[(ii-1) * batch + 1:ii*batch], training[(ii-1) * batch + 1:ii*batch,end]), opt)
        push!(losses, sum(loss.(vecsvecs[(ii-1) * batch + 1:ii*batch], training[(ii-1) * batch + 1:ii*batch,end])))
    end
end

#---------------------
occur = countmap(wordspl)
a_i = []
for i in 1:size(calib)[1]
    if isnan(calib[i,1]) == false && (calib[i,end]) == 1
        LL = .8*log2(occur[get_vector_word[calib[i,301:600]]]) + .000000001
        LLA = 1 - nn(calib[i,1:end-1])[1]^(1/LL)
        NNA = 1 - nn(calib[i,1:end-1])[1]
        #push!(a_i,  max(NNA,LLA))
        push!(a_i,  NNA)
    end
end
a_i = sample(a_i, 200, replace = false)

counter = []
    eff = []
    for ii in 1:size(testing)[1]
        if testing[ii,51:100] != zeros(50) && testing[ii,end] == 1
            pvals = []
            for i in 1:length(uni)
                #LL = .8*log2(occur[uni[i]]) + .000000001
                #LLA = (1 - nn(vcat(testing[ii,1:300], toEmbedding([uni[i]],get_word_index)))[1]^(1/LL))
                NNA = 1 - nn(vcat(testing[ii,1:300], toEmbedding([uni[i]],get_word_index)))[1]
                a = #max(NNA,LLA)
                a = NNA
                push!(pvals, sum(a_i .> a)/(length(a_i)+1))
            end
            epsilon = .7
            pred = []
                for i in 1:length(pval)
                    if pvals[i] > epsilon
                        push!(pred, uni[i])
                    end
                end
            correctword = get_vector_word[testing[ii,301:600]]
            println(length(pred), " ", correctword in pred, " ", correctword)
            push!(counter, correctword in pred)
            push!(eff, length(pred))
            println(ii / sum(testing[:,end]), " ", sum(counter)/length(counter))
        end
    end

suma = []
for i in 1:length(a)
    push!(suma,sum(a[1:i]))
end
diffa = suma .> .8
argmax(diffa)
