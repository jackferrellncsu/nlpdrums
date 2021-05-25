using CSV
using DataFrames
arr = CSV.read("/Users/mlovig/Downloads/mtsamples.csv", DataFrame)
print(arr[1,:])
