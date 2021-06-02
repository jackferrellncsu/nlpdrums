import TextAnalysis
using InvertedIndices
using StatsBase
using RecursiveArrayTools
include("data_cleaning.jl")

#Function takes in dataframe and field of interest
#Returns document term matrix
function CreateDTM(data, field)
    docCollection = []
    balancingSamps = []
    allDocs = []

    for i in eachrow(data)

        #Strip unwanted characters with custom func
        transcriptDoc = StripUnwanted(i)

        #First for loop collects items in field of interest
        if i[1] == field
            #Append to document collection
            push!(docCollection, transcriptDoc)
            push!(allDocs, transcriptDoc)
        else
            #= Collects indeces not included in field of interest=#

            push!(balancingSamps, i)
        end
        push!(allDocs, transcriptDoc)
    end
    #Second part of function takes sample of transcripts outside field of interest
    #of equal size as the first and adds them to the document collection
    rs = sample(balancingSamps, length(docCollection), replace = false)

    for j in rs
        #Strip unwanted characters with custom func
        transcriptDoc = StripUnwanted(j)

        #Append to document collection
        push!(docCollection, transcriptDoc)
    end

    #Create corpus with cleaned string documents
    crps = Corpus(docCollection)
    totalcrps = Corpus(allDocs)
    update_lexicon!(totalcrps)
    lex = lexicon(totalcrps)

    m = DtmHelper(crps, field, lex)
    m = Vec2Mat(m)

end

#Creates raw dtm and adds label as 0 or 1 at end of each row
function DtmHelper(crps, field, lex)
    matrix = []
    t = [1]
    f = [0]
    for i in crps
        a = dtv(i, lex)
        a = vec(a)
        if author(i) == field
            push!(a, 1)
        else
            push!(a, 0)
        end

        push!(matrix, a)
    end


    return matrix
end

#Takes in row of cleaned df and returns a cleaned StringDocument
function StripUnwanted(row)
    sd = TextAnalysis.StringDocument(row[3])

    #Set all to lowercase
    remove_case!(sd)

    #Strip various unwanted things from token
    #Note: commented out commands to strip numbers and such
    #Commented out spares_terms bc I dont know what it does
    prepare!(sd, strip_articles)
    prepare!(sd, strip_indefinite_articles)
    prepare!(sd, strip_definite_articles)
    prepare!(sd, strip_prepositions)
    prepare!(sd, strip_pronouns)
    prepare!(sd, strip_stopwords)
    #prepare!(sd, strip_numbers)
    prepare!(sd, strip_non_letters)
    #prepare!(sd, strip_spares_terms)
    #prepare!(sd, strip_frequent_terms)
    prepare!(sd, strip_html_tags)

    #Could also use stem function to further cut terms
    #stem!(sd)

    #Set doc title to be the name of sample
    TextAnalysis.title!(sd, row[2])

    #Set doc author to be field its from
    author!(sd, row[1])

    return sd
end

function Vec2Mat(v)
    VA = VectorOfArray(v)
    return(convert(Array, VA))
end
