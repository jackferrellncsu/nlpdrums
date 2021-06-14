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
function createCorpusText(data,pads)
   allDocs = ""
   for i in 1:length(data[:,3])
      println(i)
      allDocs = allDocs * " " * data[i,3]
   end
   open("corpus.txt","a") do io
      println(io,allDocs)
   end
end

# Cleans up data a bit more before train/test split
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
function formulateText(model, script)
   words = split(script, " ")
   vecs = zeros(length(get_vector(model,"the")))
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
