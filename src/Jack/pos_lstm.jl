using CUDA: include, collect
using JLD
using Flux
using CUDA
using Embeddings


include("brown_conf.jl")

sentences, pos_tags = brown_loading()
act_word, act_pos, new_sentences = word_masker(sentences, pos_tags)
unique_words = get_uniques(sentences)


#embtable = load_embeddings(GloVe{:en}, 6, keep_words = unique_words)
#JLD.save("posEmbs.jld", "embtable", embtable)

#embtable and useful dicts
embtable = JLD.load("Data/posEmbs.jld", "embtable")
vector_from_word = Dict(word=>embtable.embeddings[:,ii] for (ii,word) in enumerate(embtable.vocab))
word_from_vector = Dict(embtable.embeddings[:, ii] => word for (ii, word) in enumerate(embtable.vocab))

#LSTM will have forwards and backwards, each with input 300, output size of pos_tags

