using Transformers
using Lathe
using Transformers.Pretrain
using Transformers.GenerativePreTrain
using Transformers.BidirectionalEncoder
using Transformers.Datasets
using Transformers.Datasets.GLUE
using Transformers.Basic
using Flux: onehotbatch

include("../data_cleaning.jl")
include("../embeddings_nn.jl")


MED_DATA = importClean()
MED_DATA_F = filtration(MED_DATA, " Cardiovascular / Pulmonary")

#generation of training and testing data
train, test = TrainTestSplit(MED_DATA_F, .9)

#Generate labels
labels = ("Cardiovascular/Pulmonary", "Not Cardiovascular/Pulmonary")

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

#Fetch the pretrained bert embeddings
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"

vocab = Vocabulary(wordpiece) #get wordpiece vocab


#add start and separate symbol around each "sentence" (doc in our case)
markline(s1) = ["[CLS]", s1, "[SEP]"]


function prep(data)
    s1 = wordpiece.(tokenizer.(data[3]))

    sentence = markline.(s1)

    mask = getmask(sentence)
    tok = vocab(sentence)

    segment = fill!(similar(tok), 1)

    label = (data[1] .== " Cardiovascular / Pulmonary")

    return (tok = tok, segment = segment), label, mask

end

using Flux
using Flux: gradient
import Flux.Optimise: update!

clf = Chain(
    Dropout(0.1),
    Dense(768, length(labels)), logsoftmax
)

#remove masklm/nextsentence weights,
# set clf as part of classifiers
#move the result model to gpu

bert_model = gpu(
    Basic.set_classifier(bert_model,
    (
        pooler = bert_model.classifier.pooler,
        clf = clf
    )
    )
)
@show bert_model

ps = params(bert_model)
opt = ADAM(1e-4)

#define the loss
function loss(data, label, mask=nothing)
    e = bert_model.embed(data)
    t = bert_model.transformers(e, mask)
    l = Basic.logcrossentropy(
        label,
        bert_model.classifier.clf(
            bert_model.classifier.pooler(
            t[:,1,:]
            )
        )
    )
    return l
end

for i ∈ 1:10
    data, label, mask = todevice(
        prep(train[i, :])
    )

    l = loss(data, label)
    @show l


    grad = gradient(()->l, ps)

    update!(opt, ps, grad)
end
