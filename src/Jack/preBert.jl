using Transformers
using Transformers.Basic
using Transformers.Pretrain

include("../data_cleaning.jl")
include("../embeddings_nn.jl")

#=================Data loading and such===============#
field = " Cardiovascular / Pulmonary"
MED_DATA = importClean()

MED_DATA = filtration(MED_DATA, field)
MED_DATA = hcat(MED_DATA, 1*(MED_DATA[:, 1] .== field))

trainVal, test = TrainTestSplit(MED_DATA, .9)

#load bert pretrains
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)

markline(x) = ["[CLS]", x, "[SEP]"]

function preprocess(data)

end
