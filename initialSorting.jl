
using CSV
using DataFrames
using Plots
using StatsBase


filename = "mtsamples.csv"
filepath = joinpath(@__DIR__, filename)
println(filepath)

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

a = countmap(split(words," "))
b = [a[k] for k in sort(collect(keys(a)))]
bar(reverse(sort(b[1500:2000])))

function propOccur(x)
    count = 0
    for i in trans
        if x in split(i)
            count+=1
        end
    end
    return count
end

function propOccur(x,y)
    count = 0
    for i in trans
        if x in split(i) && y in split(i)
            count+=1
        end
    end
    return count
end
