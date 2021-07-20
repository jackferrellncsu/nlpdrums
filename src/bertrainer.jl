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
    using Transformers
    using Transformers.Basic
    using Transformers.Pretrain

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
        sent = sent .* " "
        push!(vectorizedSpl, vcat(sent, [String(".")]))
        vectorizedSpl[i] = filter(x->x≠"",vectorizedSpl[i])
        vectorizedSpl[i] = filter(x->x≠" ",vectorizedSpl[i])
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
function makeDataByWord(sentances, nextword, bert_model, wordpiece, tokenizer, vocab)
    sentancesVecs = []
    nextwordVecs = []
    for i in ProgressBar(1:length(sentances))

        text1 = join(sentances[i]) |> tokenizer |> wordpiece

        text = ["[CLS]"; text1; "[SEP]"]

        token_indices = vocab(text)
        segment_indices = [fill(1, length(text1)+2);]

        sample = (tok = token_indices, segment = segment_indices)

        bert_embedding = sample |> bert_model.embed
        feature_tensors = bert_embedding |> bert_model.transformers

        push!(sentancesVecs, feature_tensors[:,1])

        text1 = join(nextword[i]) |> tokenizer |> wordpiece


        text = ["[CLS]"; text1; "[SEP]";]

        token_indices = vocab(text)
        segment_indices = [fill(1, length(text1)+2);]

        sample = (tok = token_indices, segment = segment_indices)

        bert_embedding = sample |> bert_model.embed
        feature_tensors = bert_embedding |> bert_model.transformers

        push!(nextwordVecs,feature_tensors[:,1])
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

function vecvec_to_matrix(vecvec)
    dim1 = length(vecvec)
    dim2 = length(vecvec[1])
    my_array = zeros(Float32, dim1, dim2)
    for i in ProgressBar(1:dim1)
        for j in 1:dim2
                my_array[i,j] = vecvec[i][j]
        end
    end
    return my_array
end


ENV["DATADEPS_ALWAYS_ACCEPT"] = true

bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"

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

vocab = Vocabulary(wordpiece)

BERTEMB = makeDataByWord(S,N,bert_model, wordpiece, tokenizer, vocab)

SentanceEmbedings = vecvec_to_matrix(BERTEMB[1])
nextWordEmbedings = vecvec_to_matrix(BERTEMB[2])
mat = hcat(SentanceEmbedings,nextWordEmbedings)
df = DataFrame(mat)

training,extra = TrainTestSplit(df, .9)
calib,testing = TrainTestSplit(extra, .9)

training = Matrix(training)
testing= Matrix(testing)
calib = Matrix(calib)

regnet = Chain(
    Dense(768, 300, gelu),
    Dense(300, 300, gelu),
    Dense(300, 300, gelu),
    Dense(300, 300,  gelu),
    Dense(300, 768,  x -> x))
    ps = Flux.params(regnet)

function loss(x, y)
    return norm(regnet(x) - y)
end

losses = []
batch  = 1000
epochs = 50
eta = .00005
opt = RADAM(eta)

for i in 1:epochs
    training = training[shuffle(1:end), :]
    vecsvecsx = []
    for i in ProgressBar(1:size(training)[1])
        push!(vecsvecsx, training[i, 1:768])
    end
    vecsvecsy = []
    for i in ProgressBar(1:size(training)[1])
        push!(vecsvecsy, training[i, 769:end])
    end
    for ii in ProgressBar(1:Int(floor(length(vecsvecsx)/batch)))
        if ii%10 == 9
            opt = RADAM(eta)
            eta = eta*.85
        end
        L = sum(loss.(vecsvecsx[(ii-1) * batch + 1:ii*batch], vecsvecsy[(ii-1) * batch + 1:ii*batch]))
        Flux.train!(loss, ps, zip(vecsvecsx[(ii-1) * batch + 1:ii*batch], vecsvecsy[(ii-1) * batch + 1:ii*batch]), opt)
        push!(losses, L/batch)

        print("         ", L/batch)
    end
end
