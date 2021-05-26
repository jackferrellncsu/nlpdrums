
using CSV
using DataFrames

filename = "mtsamples.csv"
filepath = joinpath(@__DIR__, filename)
println(filepath)

arr = CSV.read(filepath, DataFrame)

for i in 1:4999
    if ismissing(arr[i,5]) == false
        #println(i)
        arr[i,5] = replace(arr[i,5], ".," => " ")
        arr[i,5] = replace(arr[i,5], [',',';','.',')','('] => "")
        arr[i,5] = replace(arr[i,5], r":" => ": ")
        arr[i,5] = replace(arr[i,5], r"\s\s+"=> " ")
        arr[i,5] = lowercase(arr[i,5])
    end
end

field = arr[:, 3]
trans = arr[:, 5]

for i in reverse(1:4999)
    if ismissing(trans[i]) == true
        global field = field[1:end.!=i]
        global trans = trans[1:end.!=i]
    end
end

uniField = [[],[]]


for n in 1:4966
    if field[n] ∉ uniField[[1]]
        print(field[n,:])
        push!(uniField[1], field[n,:])
        push!(uniField[2], 1)
    else
        for ii in uniField[1]
            uniField[2,findall(x->x=ii)]+= 1
        end
    end
end
