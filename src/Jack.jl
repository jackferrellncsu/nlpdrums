using TextAnalysis
using InvertedIndices
using StatsBase

#Create corpus with cleaned string documents
crps = Corpus(docCollection)
update_lexicon!(crps)

#Create DT Matrix
docTerm = DocumentTermMatrix(crps)
m = dtm(docTerm)

createDTM(data, " Bariatrics")

#Function takes in dataframe and field of interest
#Returns document term matrix
function createDTM(data, field)
    docCollection = []
    balancingSamps = []

    for i in eachrow(data)

        #First for loop collects items in field of interest
        if i[1] == field
            #Strip unwanted characters with custom func
            transcriptDoc = StripUnwanted(i)

            #Append to document collection
            push!(docCollection, transcriptDoc)
        else
            #= Collects indeces not included in field of interest=#
            push!(balancingSamps, i)
        end
    end
    #Second part of function takes sample of transcripts outside field of interest
    #of equal size as the first and adds them to the document collection
    rs = sample(balancingSamps, length(docCollection), replace = false)
end

#Takes in row of cleaned df and returns a cleaned StringDocument
function StripUnwanted(row)
    sd = StringDocument(row[3])

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
    title!(sd, row[2])

    #Set doc author to be field its from
    author!(sd, row[1])

    return sd
end
