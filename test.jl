
using CSV
using DataFrames
# arr = CSV.read("/Users/mlovig/Downloads/mtsamples.csv", DataFrame)

filename = "mtsamples.csv"
filepath = joinpath(@__DIR__, filename)
println(filepath)

arr = CSV.read(filepath, DataFrame)

for i in 1:4999
    if ismissing(arr[i,5]) == false
        println(i)
        arr[i,5] = replace(arr[i,5], ".," => " ")
        arr[i,5] = replace(arr[i,5], [',',';','.',')','('] => "")
        arr[i,5] = replace(arr[i,5], r":" => ": ")
        arr[i,5] = replace(arr[i,5], r"\s\s+"=> " ")
        arr[i,5] = lowercase(arr[i,5])
    end
end
field = arr[:, 3]
trans = arr[:, 5]
println(field[1])
println(trans[1])
