using SparseArrays
using SparseArrayKit
using JLD
using Word2Vec
using LinearAlgebra
using Statistics
using Embeddings
using Flux
using DataFrames
using Lathe.preprocess: TrainTestSplit
using Random

f = open("/Users/mlovig/Downloads/1342-0.txt", "r")
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
    corp =replace(corp, "." => " .")
    corp =replace(corp, "!" => " .")
    corp =replace(corp, "?" => " .")
    corp =replace(corp, ")" => "")
    corp =replace(corp, "(" => "")
    corp =replace(corp, "," => "")
    corp =replace(corp, "_" => "")
    corp =replace(corp, "—" => " ")
    corp =replace(corp, "-" => " ")
    corp =replace(corp, "—" => " ")
    spl = convert(Vector{String},split(corp, "."))

vectorizedSpl = []
    for i in 1:length(spl)
        sent = convert(Vector{String},split(spl[i], " "))
        push!(vectorizedSpl, append!(sent, [String(".")]))
        vectorizedSpl[i] = filter(x->x≠"",vectorizedSpl[i])
    end

    vectorizedSpl = vectorizedSpl[1:end-219]

wordspl = convert(Vector{String},split(corp, " "))
    wordspl = filter(x->(x≠"" && x≠"."),wordspl)
    wordspl = wordspl[1:findall(x -> x == ".***", wordspl)[1]-1]

uni = convert(Vector{String},unique(wordspl))
D = Dict(1:length(uni) .=> uni)

sentances = []
nextword = []
for i in 1:length(vectorizedSpl)
    println(i/length(vectorizedSpl))
    for ii in 1:length(vectorizedSpl[i])-1
        push!(sentances,vectorizedSpl[i][1:ii])
        push!(nextword,vectorizedSpl[i][ii+1])
    end
end

embtable = load_embeddings(GloVe{:en},1, keep_words=Set(uni))
JLD.save("embtable.jld", "embtable", embtable)

t = JLD.load("embtable.jld")
embtable = t["embtable"]
get_word_index = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab))
get_vector_word = Dict(embtable.embeddings[:,ii]=>word for (ii,word) in enumerate(embtable.vocab))
sentancesVecs = []
nextwordVecs = []

JLD.save("PridePrej.jld", "corpus", corp, "sentances", vectorizedSpl, "data", hcat(sentances,nextword), "embeddings", get_word_index)

for i in 1:length(sentances)
    println(i / length(sentances))
    if NaN ∉ toEmbedding(sentances[i],get_word_index) &&
        NaN ∉ toEmbedding([nextword[i]],get_word_index) &&
        zeros(300) != toEmbedding([nextword[i]],get_word_index) &&
        zeros(300) != toEmbedding(sentances[i],get_word_index)
            push!(sentancesVecs, toEmbedding(sentances[i],get_word_index))
            push!(nextwordVecs, toEmbedding([nextword[i]],get_word_index))
    end
end

bogusNextWord = []
incorrects = 10
while length(bogusNextWord) < incorrects*length(sentancesVecs)
    println(length(bogusNextWord) / (incorrects*length(sentancesVecs)))
    W = toEmbedding([uni[rand(1:length(uni))]], get_word_index)
    if NaN ∉ W && zeros(300) != W
        push!(bogusNextWord, W)
    end
end
combo = vcat.(sentancesVecs,nextwordVecs)
for i in 1:incorrects
    combo = vcat(combo, vcat.(sentancesVecs, bogusNextWord[(i-1)*length(sentancesVecs)+1:(i)*length(sentancesVecs)]))
end

resp = Int.(vcat(ones(Int(length(combo)/(incorrects+1))),zeros(Int(incorrects*length(combo)/(incorrects+1)))))
mat = vecvec_to_matrix(vcat.(combo,resp))
df = DataFrame(mat)

training,testing = TrainTestSplit(df, .8)
training,calib = TrainTestSplit(training, .8)

training = Matrix(training)
testing= Matrix(testing)
calib = Matrix(calib)

nn = Chain(
    Dense(100, 50, gelu),Dense(50,10,gelu),Dense(10, 1, (x->σ(x)))
    )

opt = RADAM()
ps = Flux.params(nn)

function loss(x, y)
    L = Flux.Losses.binarycrossentropy(nn(x),y)
    if isnan(L)
        return 0
    else
        return L
    end
end

for i in 1:2
    training = training[shuffle(1:end), :]
    vecsvecs = []
    for i in 1:size(training)[1]
        push!(vecsvecs, training[i, 1:end-1])
    end
    for ii in 1:Int(floor(length(vecsvecs)/1000))
        println(i, " ", ii)
        Flux.train!(loss, ps, zip(vecsvecs[(ii-1) * 1000 + 1:ii*1000], training[(ii-1) * 1000 + 1:ii*1000,end]), opt)
    end
end

#---------------------
a_i = []
for i in 1:size(calib)[1]
    if isnan(calib[i,1]) == false && (calib[i,end]) == 1
        push!(a_i, 1 - nn(calib[i,1:end-1])[1])
    end
end

newa_i = []
for x in a_i
    if rand() < .01
        push!(newa_i, x)
    end
end
a_i = newa_i

text = "sir william and lady"
PSEmbed = convert(Vector{String},split(text, " "))
PSEmbed = filter(x->(x≠""),PSEmbed)
PSEmbed = toEmbedding(PSEmbed,get_word_index)
pval = []
for i in 1:length(uni)
    println(i/length(uni))
    a = 1 - nn(vcat(PSEmbed,toEmbedding([uni[i]], get_word_index)))[1]
    push!(pval, sum(a_i .> a)/(length(a_i)+1))
end

epsilon = .2
wordpred = []
    for i in 1:length(pval)
        if pval[i] > epsilon
            push!(wordpred, D[i])
        end
    end
    println("")
    println(length(wordpred) / length(pval))
    println(length(wordpred) / length(pval))

println(wordpred)

correct = []
for i in 1:size(training)[1]
    println(i/size(training)[1])
    push!(correct, (nn(training[i,1:600])[1] > .5) == training[i,end])
end

counter = []
eff = []
for ii in 1:size(testing)[1]
    if testing[ii,51:100] != zeros(50) && testing[ii,end] == 1
        pvals = []
        for i in 1:length(uni)
            a = 1 - nn(vcat(testing[ii,1:50], toEmbedding([uni[i]],get_word_index)))[1]
            push!(pvals, sum(a_i .> a)/(length(a_i)+1))
        end
        epsilon = .5
        pred = []
            for i in 1:length(pval)
                if pvals[i] > epsilon
                    push!(pred, uni[i])
                end
            end
        correctword = get_vector_word[testing[ii,51:100]]
        println(length(pred), " ", correctword in wordpred, " ", correctword)
        push!(counter, correctword in wordpred)
        push!(eff, length(pred))
        println(ii / sum(testing[:,end]), " ", sum(counter)/length(counter))
    end
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

function vecvec_to_matrix(vecvec)
    dim1 = length(vecvec)
    dim2 = length(vecvec[1])
    my_array = zeros(Float32, dim1, dim2)
    for i in 1:dim1
        println(i/dim1)
        for j in 1:dim2
            my_array[i,j] = vecvec[i][j]
        end
    end
    return my_array
end
