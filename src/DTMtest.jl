include("DTMCreation.jl")
include("data_cleaning.jl")
include("PCA.jl")
using WordCloud



data = importClean()
dtb = CreateDTM(data, " Cardiovascular / Pulmonary")
dtmnew = dtm'
new = PCA(dtm[:, 1:end - 1])
