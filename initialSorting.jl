
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

#=Removing punctuation =#
for i in 1:size(arr)[1]
    if ismissing(arr[i,5]) == false
        #println(i)
        arr[i,5] = replace(arr[i,5], ".," => " ")
        arr[i,5] = replace(arr[i,5], [',',';','.',')','(', '!', '+', '{', '}',
                                      '[', ']', '-', '+', '_', '~', ''', '"', '*',
                                      '?', '<', '>', '%', '$'] => "")
        arr[i,5] = replace(arr[i,5], r":" => ": ")
        arr[i,5] = replace(arr[i,5], r"\s\s+"=> " ")
        arr[i,5] = lowercase(arr[i,5])
    end
end

#=Extracting columns with valuable info, including medical field
and actual transcriptions =#
field = arr[:, 3]
trans = arr[:, 5]
name = arr[:,4]

#=Removing rows with missing transcripts=#
for i in reverse(1:4999)
    if ismissing(trans[i]) == true
        global field = field[1:end.!=i]
        global trans = trans[1:end.!=i]
    end
end

#=Creating frequency chart for medical fields =#
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

#=Creating frequency chart for each word =#
words = " "

for i in 1:length(trans)
    global words = string(words," ",trans[i])
end

uniWords = unique(split(words," "))
uniSpecWords = []


plotlyjs()

a = countmap(split(words," "))
b = [a[k] for k in sort(collect(keys(a)))]
bar(reverse(sort(b[500:5000])))

#Finding out transcript bodies aren't unique.
count(x -> (x < 1000 && x > 5), b)

length(unique(name))


runexample(:alice)
showexample(:alice)

summarystats(b)
