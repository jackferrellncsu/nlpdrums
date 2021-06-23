
using Transformers
using Transformers.Basic
using Transformers.Pretrain
using Transformers.BidirectionalEncoder
using Lathe

using Flux
using Flux: onehotbatch, gradient
import Flux.Optimise: update!
using WordTokenizers

include("../data_cleaning.jl")
include("../embeddings_nn.jl")


Pkg.activate(".")
#=================Data loading and such===============#
field = " Cardiovascular / Pulmonary"
MED_DATA = importCleanSent()

MED_DATA = filtration(MED_DATA, field)
MED_DATA = hcat(MED_DATA, 1*(MED_DATA[:, 1] .== field))

trainVal, test = TrainTestSplit(MED_DATA, .9)

#trainVal refers to training/validation, will need to split again at some point
trainVal = convert(Matrix, trainVal)
test = convert(Matrix, test)

#Important part, needs to be iterated through this format for some reason
trainValForm = [selectdim(trainVal, 2, i) for i = (3, 4)]
test = [selectdim(test, 2, i) for i = (3, 4)]

const labels = ("0", "1")

#Need to convert data into channels
trainValForm = dataset_med(trainVal, "med")

getSplit.(trainVal[:, 3])
function getSplit(str::String)
    splits = Vector{String}()
    if length(split(str)) > 200
        splits = split(str, '.')
    else
        splits = [str]
    end
    return splits
end

split(MED_DATA[1, 3], '.')


#=================Loading Bert Stuff==========================================#
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"

vocab = Transformers.Vocabulary(wordpiece)

markline(sen) = ["[CLS]"; sen; "[SEP]"]

function preprocess(batch)
    sentence = markline.(wordpiece.(tokenizer.(batch[1])))
    mask = getmask(sentence)
    tok = vocab(sentence)
    segment = fill!(similar(tok), 1)



    label = onehotbatch(batch[2], labels)

    return (tok=tok, segment = segment), label, mask
end

p = []
for i ∈ 1:10
    batch = get_batch(trainValForm, 2)
    data, label, mask  = todevice(
        preprocess(batch)
    )
    bert_embedding = data |> bert_model.embed
    push!(p, bert_embedding)


    feature_tensors = bert_embedding |> bert_model.transformers



end



#======================Training and testing model==============================#
clf = gpu(Chain(
        Dropout(0.1),
        Dense(size(bert_model.classifier.pooler.W, 1), length(labels)),
        logsoftmax
))



bert_model = gpu(
    Basic.set_classifier(bert_model,
        (
            pooler = bert_model.classifier.pooler,
            clf = clf
        )
    )
)

ps = params(bert_model)
opt = ADAM(1e-4)

function loss(data, label, mask=nothing)
    e = bert_model.embed(data)
    t = bert_model.transformers(e, mask)
    l = Basic.logcrossentropy(
        label,
        bert_model.classifier.clf(
            bert_model.classifier.pooler(
                t[:, 1, :]
            )
        )
    )
    return l
end

for i ∈ 1:10
    batch = get_batch(trainValForm, 2)
    batch === nothing && break

    data, label, mask = todevice(
        preprocess(batch)
    )
    l = loss(data, label, mask)
    @show l
    grad = gradient(()->l, ps)
    update!(opt, ps, grad)
end

#=======================Rewriting things I need from src code==================#
function get_channels_j(::Type{T}, n; buffer_size = 0) where T
    Tuple([Channel{T}(buffer_size) for i in 1:n])
end

function dataset_med(data, type)
    needed_fields = type == "med" ? (3,4) : (1, 2)

    rds = get_channels_j(String, length(needed_fields); buffer_size = size(data)[1])

    task = @async begin
        for r in eachrow(data)
            for (i, j) ∈ enumerate(needed_fields)
                put!(rds[i], string(r[j]))
            end
        end
    end
    for rd ∈ rds
        bind(rd, task)
    end

    return rds
end

function get_split()
    body
end
