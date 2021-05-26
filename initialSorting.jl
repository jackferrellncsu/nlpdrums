
using CSV
using DataFrames
using Plots
using StatsBase
using PlotlyJS

filename = "mtsamples.csv"
filepath = joinpath(@__DIR__, filename)


arr = CSV.read(filepath, DataFrame)

for i in 1:size(arr)[1]
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
name = arr[:,4]

for i in reverse(1:4999)
    if ismissing(trans[i]) == true
        global field = field[1:end.!=i]
        global trans = trans[1:end.!=i]
    end
end

uniName = unique(field)
uniFeq = zeros(length(uniName))


for i in 1:length(field)
    for ii in 1:length(uniName)
        if field[i] == uniName[ii]
            global uniFeq[ii] += 1
        end
    end
end

bar(uniName,uniFeq,yticks = :all,orientation = :h)

words = " "

for i in 1:length(trans)
    global words = string(words," ",trans[i])
end

uniWords = unique(split(words," "))
uniSpecWords = []

#=
commonWords = CSV.read("/Users/mlovig/Downloads/4000-most-common-english-words-csv.csv", DataFrame)

for i in uniWords
    if string(i) ∉ commonWords[:,1]
        push!(uniSpecWords, i)
    end
end

=#

plotlyjs()

a = countmap(split(words," "))
b = [a[k] for k in sort(collect(keys(a)))]
bar(reverse(sort(b[500:5000])))

count(x -> (x < 1000 && x > 5), b)

length(unique(name))

struct name
    fields
end # struct
