using Word2Vec
using DataFrames
using Languages
using Lathe.preprocess: TrainTestSplit

include("data_cleaning.jl")

export createCorpusText, filtration, formulateText

# ---------------------------------------------------------------
# ------------------------- Functions ------------------------- #
# ---------------------------------------------------------------

# Creates corpus text file
# choice = 2; padding after any doc
# choice = 1; creates corpus with padding inbetween medical fields
# choice = 0; no padding
function createCorpusText(data, choice)
   allDocs = ""
   thePad = ""
   for i in 1:3000
      thePad = thePad * " randomWordNow"
   end
   for i in 1:length(data[:, 3])
      println(i)
      if choice == 1
         if i != 1
            if data[i, 1] != data[i-1, 1]
               println("This is a seperation")
               allDocs = allDocs * thePad * " " * data[i, 3]
            else
               allDocs = allDocs * " " * data[i, 3]
            end
         end
      elseif choice == 0
         allDocs = allDocs * " " * data[i, 3]
      elseif choice == 2
         allDocs = allDocs * thePad * " " * data[i, 3]
      end
   end
   open("corpus.txt","a") do io
      println(io,allDocs)
   end
end

# Cleans up data a bit more before train/test split, samples data 50/50
function filtration(df, field)
   indexes = []
   for i in 1:length(df[:,1])
      if df[i,1] == field
         push!(indexes,i)
      else
         if rand() < sum(df[:,1].==field)/(length(df[:,1]) - sum(df[:,1].==field))
            push!(indexes,i)
         end
      end
   end
   return df[indexes,:]
end

# Formulates the text for the creation of the embeddings matrix
# Turns documents to vectors
function formulateText(model, script)
   words = split(script, " ")
   vecs = zeros(length(get_vector(model,vocabulary(model)[1])))
   counter = 0
   for i in words[1:end]
      if i in vocabulary(model) && i ∉ stopwords(Languages.English())
         vecs = vecs .+ get_vector(model,i)
          #&& i ∉ stopwords(Languages.English())
         counter += 1
      end
   end
   return vecs ./ counter
end

# Takes in multinomial probabilities and creates
# a one hot vector
function classField(vec)
   out = zeros(length(vec))
   out[argmax(vec)] = 1
   return out
end

# matrix to vector of vector
function getVector(mat::AbstractMatrix{T}) where T
   len, wid = size(mat)
   B = Vector{T}[Vector{T}(undef, wid) for _ in 1:len]
   for i in 1:len
      B[i] .= mat[i, :]
   end
   return B
end
