using CSV
using Random

Random.seed!(1234)

words = ['A':'Z';]

#Setting constant parameters for data creation
NUMDOCS = 50
MIN_DOC_LENGTH = 25
MAX_DOC_LENGTH = 50

P_HOT_DOC = 0.5
P_KEY_PHRASE_1 = 0.05
P_KEY_PHRASE_0 = 0.01




synthdocs = Vector{Tuple{Int8, String}}()
num_hot = 0
for i in 1:NUMDOCS

    #initialize current doc being created to not be one of interest
    currentdoc = ""
    hotdoc = 0

    #Pick a random number between max and min doc length
    currentdoclength = rand(MIN_DOC_LENGTH:MAX_DOC_LENGTH, 1)[1]

    #decides whether or not current doc will be doc of interest
    #based on chosen probability
    if rand() < P_HOT_DOC
        hotdoc = 1

        for j in 1:currentdoclength
            if rand() < P_KEY_PHRASE_1
                #put space at end of anything you concat to doc
                currentdoc *= "A M D "
            else
                currentdoc *= String(rand(words, 1)) * " "
            end
        end
    else
        for j in 1:currentdoclength
            currentdoc *= String(rand(words, 1)) * " "
        end
    end

    #Alternate for embeddings

    num_hot += hotdoc
    println(i)
    push!(synthdocs, (hotdoc, currentdoc))

end

#Write synthetic data to txt file

open("src/Jack/synthdata.txt", "w") do io
    for d in synthdocs
        write(io, string(d[1]) * " ")
        write(io, d[2])
        write(io, '\n')
    end
end
