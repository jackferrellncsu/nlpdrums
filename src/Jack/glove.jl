using Glowe
using Lathe
using Embeddings
using LinearAlgebra

include("../data_cleaning.jl")
include("../embeddings_nn.jl")

MED_DATA = importClean()

field = " Cardiovascular / Pulmonary"
sub = filtration(MED_DATA, field)

trainVal, test = TrainTestSplit(sub, 0.9)

train, validate = TrainTestSplit(trainVal, 0.9)

#Load embeddings
embtable = load_embeddings(GloVe{:en}, 3, max_vocab_size = 50000)
#create dict for embeddings
const get_word_index = Dict(word=>ii for (ii, word) in enumerate(embtable.vocab))
const vec_length = length(embtable.embeddings[:, get_word_index["the"]])

#Get embeddings for each set
trainEmbs = SampleEmbeddings(train, vec_length)
testEmbs = SampleEmbeddings(test, vec_length)
valEmbs = SampleEmbeddings(validate, vec_length)

#Get classifications for train/val/test
classTrain = train[:, 1] .== field
classTrain = classTrain * 1.0

classTest = test[:, 1] .== field
classTest = classTest * 1.0

classVal = validate[:, 1] .== field
classVal = classVal * 1.0

batchsize_custom = 100
trainDL = Flux.Data.DataLoader((trainEmbs, classTrain'),
                                batchsize = batchsize_custom,
                                shuffle = true)
valDL = Flux.Data.DataLoader((valEmbs, classVal'))

nn = Chain(Dense(200, 100, relu),
            Dense(100, 10, relu),
            Dense(10, 1, x->σ.()))

opt = RADAM()

#finish model tmrw





#returns sum of embeddings to feed into nn
function SampleEmbeddings(df, vec_size)
    embed = 0
    embed_mat = Matrix{Float64}(I, vec_size, length(eachrow(df)))
    for (i, r) in enumerate(eachrow(df))
        doc = split(r[3])
        embed = getEmbedding(doc[1])
        for d in doc[2:end]
            embed += getEmbedding(d)
        end
        embed_mat[:, i] = embed
    end

    return embed_mat
end

function getEmbedding(word)
    if word ∈ keys(get_word_index)
        ind = get_word_index[word]
        emb = embtable.embeddings[:, ind]
    else
        vec_length = length(embtable.embeddings[:, get_word_index["the"]])
        emb = zeros(vec_length)
    end
    return emb
end
