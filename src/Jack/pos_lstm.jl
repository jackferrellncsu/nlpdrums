using CUDA: include, collect
using JLD
using Flux
using CUDA
using Embeddings


include("brown_conf.jl")

sentences, pos_tags = brown_loading()
act_word, act_pos, new_sentences = word_masker(sentences, pos_tags)
unique_words = get_uniques(sentences)


embtable = load_embeddings(GloVe{:en}, 6, keep_words = unique_words)