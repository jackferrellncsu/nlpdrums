using CSV
using DataFrames
using Plots
using StatsBase
using PlotlyJS
using WordCloud

#=Reading csv file =#
filename = "mtsamples.csv"
filepath = joinpath(@__DIR__, filename)
arr = CSV.read(filepath, DataFrame)
