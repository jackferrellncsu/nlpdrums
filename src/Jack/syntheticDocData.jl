using CSV
using Random
using Word2Vec

Random.seed!(1234)

#Setting constant parameters for data creation
struct SynthParams
    NUMDOCS
    MIN_DOC_LENGTH
    MAX_DOC_LENGTH

    P_HOT_DOC
    P_KEY_PHRASE_1
    P_KEY_PHRASE_0
    DIST1
    DIST2
end

words = ['A':'Z';]

words2 = ['A':'E';]

wordsdist1 = CreateWordDistibution(words)
wordsdist2 = CreateWordDistibution(words)

X = SynthParams(50, 25, 25, 0.5, 0.05, 0.01, wordsdist1, wordsdist2)

Y = createSynthData(X, words)
WriteData("synthdata.txt", "synthcrps.txt", Y)




function createSynthData(X, vocab)
    synthdocs = Vector{Tuple{Bool, String}}()
    for i in 1:X.NUMDOCS
        currentDoc = ""
        hotdoc = rand() < X.P_HOT_DOC
        currWord = ""

        currentDocLength = rand(X.MIN_DOC_LENGTH:X.MAX_DOC_LENGTH, 1)[1]

        j = 1
        while j <= currentDocLength
            if j == 1
                currWord = rand(vocab, 1)[1]
                currentDoc *= currWord * " "
            else
                prevWord = currWord

                prob = rand()
                if hotdoc
                    currWord = sampleFromWords(prob, prevWord, X.DIST1, vocab)
                else
                    currWord = sampleFromWords(prob, prevWord, X.DIST2, vocab)
                end

                currentDoc *= currWord * " "
            end
            j += 1
        end
        push!(synthdocs, (hotdoc, currentDoc))
    end
    return synthdocs
end

function WriteData(filename, crpsfile, data)
    open(filename, "w") do io
        for d in data
            write(io, string(1*d[1]) * " ")
            write(io, d[2])
            write(io, '\n')
        end
    end


    open(crpsfile, "w") do io
        for d in data
            write(io, d[2])
            write(io, '\n')
        end
    end
end


function sampleFromWords(p, prev, distDict, vocab)
    chosenNext = 'a'
    lower = 0

    for c in vocab

        if p < distDict[prev][c] + lower && distDict[prev][c] != 0
            return c
        else
            lower += distDict[prev][c]
        end
    end
    return chosenNext
end



function CreateWordDistibution(vocab, strong)
    dists = Dict{Char, Dict{Char, Float64}}()
    for w in vocab
        dists[w] = Dict(w=>0)

        
        probs = rand(length(vocab) - 1)
        probs = probs ./ sum(probs)

        i = 1
        for x in vocab
            if x == w
                dists[w][x] = 0
            else
                dists[w][x] = probs[i]
                i += 1
            end
        end
    end
    return dists
end

word2vec("synthcrps.txt", "synthvectors.txt")

m = wordvectors("synthvectors.txt")
similarity(m, "A", "Z")
