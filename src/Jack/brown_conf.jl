using DataFrames
using Random
using CSV
using StatsBase
# ------------------------ Data Cleaning ------------------------ #
"""
    brown_loading()

Requires no input, simply loads brown corpus and returns the sentences and tags.
"""
function brown_loading()
    brown_df = CSV.read("Data/brown.csv", DataFrame)
    sentences = brown_df[5]
    tags = brown_df[6]
    sentences = split.(sentences, " ")
    tags = split.(tags, " ")

    return sentences, tags
end

"""
   word_masker(sentences, tags)

Takes in sentences and their part of speech tags, masks a random word in each sentence.
Returns the actual word, its position, and newly created vectors with masks.
"""
function word_masker(sentences, tags)

    act_word = []
    act_pos = []
    for i in 1:length(sentences)
        samp = sample(1:length(sentences[i]))
        push!(act_word, sentences[i][samp])
        push!(act_pos, tags[i][samp])
        sentences[i][samp] = "/MASK/"
    end
    return act_word, act_pos, sentences
end

"""
    get_uniques(sentences)

Returns all unique words in corpus
"""
function get_uniques(sentences)
    total_uniques = Set(sentences[1])
    for s in sentences[1:end]
        union!(total_uniques, Set(s))
    end

    return collect(total_uniques)
end

