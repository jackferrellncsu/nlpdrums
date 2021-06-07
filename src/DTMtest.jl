include("DTMCreation.jl")
include("data_cleaning.jl")
using WordCloud

data = importClean()
dtm = CreateDTM(data, " Bariatrics")
