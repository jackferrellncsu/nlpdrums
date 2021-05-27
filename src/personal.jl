
using CSV
using DataFrames
using Plots
using StatsBase

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

arr = CSV.read("/Users/mlovig/Downloads/mtsamples.csv", DataFrame)

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


#Removing Unwanted Labels
remove = [" IME-QME-Work Comp etc.", " Letters", " Office Notes", " SOAP / Chart / Progress Notes", " Surgery"," Pain Management", " Discharge Summaries", " Radiology", " Neurosurgery", " Consult - History and Phy.", " Consult - History and Phy.", " Emergency Room Reports", " Discharge Summary"]
for i in reverse(1:length(field))
    if field[i] in remove
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
wordFreq = zeros(length(uniWords))

#Converting Countmap into two arrays

a = countmap(split(words," "))
tup = collect(a)
labels = []
counts = []
for i in 1:length(tup)
    push!(labels,tup[i][1])
    push!(counts,tup[i][2])
end

low = 75
up = 50
filterLabels = []
for i in 1:length(labels)
    if 75 >=counts[i] >=50
        push!(filterLabels,labels[i])
    end
end

filterTrans = []
filterField = []
match = " Neurology"
for i in 1:length(trans)
    if field[i] == match
        push!(filterTrans,trans[i])
        push!(filterField, 1)
    end
end
temp = copy(filterField)
c = length(temp)
n=0
while n <= c
    ra = rand(1:length(field))
    if field[ra]!=match
        push!(filterTrans,trans[ra])
        push!(filterField, 0)
        n+=1
    end
end

bigM = zeros(length(filterTrans),length(filterLabels))
for i in 1:size(bigM)[1]
    for ii in 1:size(bigM)[2]
        bigM[i,ii] = occursin(filterLabels[ii],filterTrans[i]) + 1 - 1
    end
end

a = countmap(split(words," "))
b = [a[k] for k in sort(collect(keys(a)))]
bar(reverse(sort(b[1500:2000])))
