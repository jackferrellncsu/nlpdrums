using CSV
using Random
using Word2Vec

Random.seed!(1234)

struct SynthParams
    NUMDOCS::Int16
    MIN_DOC_LENGTH::Int16
    MAX_DOC_LENGTH::Int16

    P_HOT_DOC::Float16
    P_KEY_PHRASE_1::Float16
    P_KEY_PHRASE_0::Float16
    DIST1
    DIST2
end

words = ['A':'Z';]

words2 = ['A':'E';]

wordsdist1 = CreateWordDistibution(words, ['A', 'Z'])
wordsdist2 = CreateWordDistibution(words, ['A', 'Z'])

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


#input list of chars for strong
function CreateWordDistibution(vocab, strong)
    dists = Dict{Char, Dict{Char, Float64}}()

    luckyword = strong[1]
    while luckyword in strong
        luckyword = rand(vocab, 1)[1]
    end
    print(luckyword)
    for w in vocab
        dists[w] = Dict(w=>0)

        probs = rand(length(vocab) - 1)


        if w in strong
            ind = findall(x -> x == luckyword, vocab)[1]

            probs[ind] = 6.0

        end

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

word2vec("synthcrps.txt", "synthvectors.txt", window = 2)

m = wordvectors("synthvectors.txt")
similarity(m, "A", "Z")
