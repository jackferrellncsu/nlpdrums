include("DTMCreation.jl")
include("data_cleaning.jl")
include("PCA.jl")
using WordCloud
using MultivariateStats



data = importClean()
dtmi = CreateDTM(data, " Cardiovascular / Pulmonary")
d = 1.0.*dtmi[1:end - 1, :]

M = MultivariateStats.fit(PCA, d; maxoutdim=100)

diagm(append!(svd(dtmi[1:end - 1, 1], full = true).S, zeros(1,2485-742)))
