using Word2Vec

numwords = 5
    samples = 10000

#Creating the Master Lookup
markovNuet = makeMarkov(numwords,1,1)

#Creating True Lookup
markovTrue = copy(markovNuet)
    #Adjusting the occurance of the word 6
    markovTrue[:,5] ./= 100000000
    #readjusting the weights
    markovTrue = fixMarkov(markovTrue)
    #Converting to CDF for estimation
    markovCDFTrue = toCDF(markovTrue)

#Creating True Samples
trueResp = []
    for i in 1:samples
        push!(trueResp,
            createSentance(markovCDFTrue,rand(10:20),true,false))
    end

#creating false lookup
markovFalse = copy(markovNuet)
    #Adjusting
    markovFalse[:,30:40] ./= 100
    #Fixing Sizes
    markovFalse = fixMarkov(markovFalse)
    #Creating CDF
    markovCDFFalse = toCDF(markovFalse)

#creating false samples
falseResp = []
    for i in 1:samples
        push!(falseResp,
            createSentance(markovCDFFalse,rand(10:20),true, false))
    end

#Creating the dataset
totalResp = []
append!(totalResp, trueResp)
append!(totalResp, falseResp)

#creating the classifiers
classifier = ones(samples)
append!(classifier, zeros(samples))

corpus = ""
for i in totalResp
    corpus = corpus * i
end

open("corpus.txt","a") do io
  println(io,corpus)
end

Vlength,window = [5,15]

word2vec("corpus.txt", "vectors.txt", size = Vlength, verbose = true,
            window = window, negative = 10, min_count = 0)
   M = wordvectors("vectors.txt", normalize = true)

   rm("vectors.txt")

rm("corpus.txt")

cosine_similar_words(M,"808")

#--------------------------------------------------------------

#=This function randomly creates a markov
 matrix with a numWeights of more fequent words
 with a larger weight of pWeight
 =#

function makeMarkov(numwords ,numWeight, pWeight)
    markov = zeros(numwords,numwords)
    for i in 1:size(markov)[1]
        for ii in 1:size(markov)[2]
            goods = sample(1:numwords, numWeight, replace = false)
            if i != ii
                if ii in goods
                    markov[i,ii] = pWeight*rand(Float32)
                else
                    markov[i,ii] = rand(Float32)
                end
            else
                markov[i,ii] = 0
            end
        end
        rowsum = sum(markov[i,:])
        for ii in 1:size(markov)[2]
            markov[i,ii] = markov[i,ii]/rowsum
        end
    end
    return markov
end


#This function will fix the markov after edits
function fixMarkov(markov)
    for i in 1:size(markov)[1]
        rowsum = sum(markov[i,:])
        for ii in 1:size(markov)[2]
            markov[i,ii] = markov[i,ii]/rowsum
        end
    end
    return markov
end

#=this function puts the markov in CDF for
 for other methods, DO NOT edit CDF Markov Matrixs
=#

function toCDF(markov)
    markovCDF = zeros(numwords,numwords)
    for i in 1:size(markov)[1]
        for ii in 1:size(markov)[2]
            markovCDF[i,ii] = sum(markov[i,1:ii])
        end
    end
    return markovCDF
end

#Creates a single sentance from a markov sentance
#It can return a string or a vector
function createSentance(markovCDF, sentanceSize, returnText, evenStart)

    #=These condtionals decide if we
    start with a even distribution or
    with a randomly selected distribution based
    on a "ghost" letter
    =#
    if evenStart
        sentance = [rand(1:numwords)]
    else
        sentance = [getnext(markovCDF[rand(1:size(markovCDF)[1]),:],rand())]
    end

    for i in 1:sentanceSize
        append!(sentance, getnext(markovCDF[sentance[end],:],rand()))
    end

    if returnText == false
        return sentance
    else
        text = ""
        for i in sentance
            text = text * string(i)* " "
        end
        return text
    end
end

#Auxilary function to get next word
function getnext(vec,num)
    for i in 1:length(vec)-1
        if num < vec[1]
            return 1
        end
        if i > 1 && num > vec[i-1] && num < vec[i]
            return i
        end
    end
    return length(vec)
end
