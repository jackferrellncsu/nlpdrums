include("DTMCreation.jl")
include("data_cleaning.jl")
include("PCA.jl")
using MultivariateStats
using Lathe
using DataFrames
using Plots;
using GLM
using StatsBase
using MLBase
using ROCAnalysis
using CSV

data = importClean()
dtmi = CreateDTM(data, " Cardiovascular / Pulmonary")
floatDtmi = 1.0*dtmi

M = fit(PCA, floatDtmi; maxoutdim = 100)
reducedMat = transform(M, floatDtmi)
