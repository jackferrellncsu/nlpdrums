using SparseArrays
using SparseArrayKit
using JLD
using Word2Vec
using LinearAlgebra
using Statistics
using Embeddings
using Flux

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
corp =replace(corp, "," => "")
corp =replace(corp, "_" => "")
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
D = Dict(uni .=> 1:length(uni))

sentances = []
nextword = []
for i in 1:length(vectorizedSpl)
    println(i/length(vectorizedSpl))
    for ii in 1:length(vectorizedSpl[i])-1
        push!(sentances,vectorizedSpl[i][1:ii])
        push!(nextword,vectorizedSpl[i][ii+1])
    end
end

embtable = load_embeddings(GloVe{:en},4, keep_words=Set(uni))
JLD.save("embtable.jld", "embtable", embtable)

t = JLD.load("embtable.jld")
embtable = t["embtable"]
get_word_index = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab))

sentancesVecs = []
nextwordVecs = []

JLD.save("PridePrej.jld", "corpus", corp, "sentances", vectorizedSpl, "data", hcat(sentances, nextword), "embtable", embtable)
for i in 1:length(sentances)
    push!(sentancesVecs, toEmbedding(sentances[i],get_word_index))
    push!(nextwordVecs, toEmbedding([nextword[i]],get_word_index))
end

nn = Chain(
    Dense(300, 500, gelu),Dense(500, 500, gelu),
    Dense(500, 500, gelu),Dense(500, 300)
    )
opt = RADAM()
ps = Flux.params(nn)

function loss(x, y)
  if rand() <= .05
     return norm(nn(x) - y)
  else
     return 0
  end
end

for i in 1:1000
    println(i)
    Flux.train!(loss, ps, zip(sentancesVecs, nextwordVecs), opt)
end
#---------------------

desiredWord = "the"
ind,vals = toVecs(getValues(S, [get(D, desiredWord, 0), -1]))
vectors = []
size = 0
for i in 1:length(vals)
    push!(vectors, vals[i] * get_vector(M, uni[ind[i]]))
    size += vals[i]
end

avrVector = sum(vectors)./size

a_is = []
for i in 1:length(vals)
    println(i/length(vals))
    if uni[i] in vocabulary(M) && get(S, [get(D, desiredWord, 0), get(D, uni[i], 0)],0 ) != 0
        for ii in 1:toVecs(getValues(S, [get(D, desiredWord, 0), get(D, uni[i], 0)]))[2][1]
            push!(a_is, norm(avrVector - get_vector(M, uni[i])))
        end
    end
end

pval = Vector{Float64}(undef,0)
    words = []
    for i in 1:length(uni)
        println(i / length(uni))
        if uni[i] in vocabulary(M)
            push!(words, uni[i])
            push!(pval,sum(a_is .>= norm(avrVector - get_vector(M, uni[i]))) / (length(a_is) + 1))
        end
    end


epsilon = .95
wordpred = []
    for i in 1:length(pval)
        if pval[i] > epsilon
            push!(wordpred, words[i])
        end
    end
    println("")
    println(length(wordpred) / length(pval))
    println(length(wordpred) / length(pval))
    #println(wordpred)

function toEmbedding(words, Embeddings)
    V = zeros(length(get(Embeddings,"the",0)))
    default = zeros(length(get(Embeddings,"the",0)))
    for x in words
        V += get(Embeddings,x,default)
    end
    return convert(Vector{Float32},V)
end
