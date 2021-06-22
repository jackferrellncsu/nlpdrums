using Transformers
using Transformers.Pretrain
using Transformers.GenerativePreTrain
using Transformers.BidirectionalEncoder
using Transformers.Datasets
using Transformers.Datasets.GLUE
using Transformers.Basic
using Flux: onehotbatch

include("../data_cleaning.jl")

get_labels(GLUE.QNLI())


MED_DATA = importClean()

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

#Messing with batching
datas = dataset()

bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"

vocab = Vocabulary(wordpiece) #get wordpiece vocab

#get labels
labels = ("Cardiovascular/Pulmonary", "Not Cardiovascular/Pulmonary")
z = MED_DATA[1, 3] |> tokenizer |> wordpiece

text = Vector{String}()

for (i, r) in enumerate(eachrow(MED_DATA))
    next = r[3] |> tokenizer |> wordpiece
    if r[1] == " Cardiovascular / Pulmonary"
        next = preprocess_field(next)
    else
        next = preprocess(next)
    end
    append!(text, next)
    println(i)
end



preprocess_field(x) = ["[CLS]", x..., "[SEP]"]
preprocess(x) = [x..., "[SEP]"]

function get_batch_custom(cs, n=1)
    res = Vector(undef, n)
    for (i, xs) ∈ enumerate(zip(cs...))
        res[i] = xs
        i >= n && break
    end
    isassigned(res, n) ? batched(res) : nothing
end
