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

uniName = unique(field)
uniFeq = zeros(length(uniName))

#Creating new arrays for field and transcript frequency
for i in 1:length(field)
    for ii in 1:length(uniName)
        if field[i] == uniName[ii]
            global uniFeq[ii] += 1
        end
    end
end

bar(uniName,uniFeq,yticks = :all,orientation = :h)


#Combining all the transcripts in one large string
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

#Defining Cutoff for word counts
low = 950
up = 1100
filterLabels = []
for i in 1:length(labels)
    if up >=counts[i] >=low
        push!(filterLabels,labels[i])
    end
end

#Matching Transcripts with the fields
filterTrans = []
filterField = []
match = " Neurology"
for i in 1:length(trans)
    if field[i] == match
        push!(filterTrans,trans[i])
        push!(filterField, 1)
    end
end

#Selecting without replacement from non-true sources
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

#Creating sparse Matrix
bigM = zeros(length(filterTrans),length(filterLabels))
for i in 1:size(bigM)[1]
    for ii in 1:size(bigM)[2]
        bigM[i,ii] = occursin(filterLabels[ii],filterTrans[i]) + 1 - 1
    end
end

#Creating fequency chart
a = countmap(split(words," "))
b = [a[k] for k in sort(collect(keys(a)))]
bar(reverse(sort(b[1500:2000])))

#Creating Heatmap of selection of words
heatWords = ["happy", "abscess", "pain", "chest", "lungs", "heart", "brain", "x-ray", "surgery", "kidney", "bone", "spine", "distress", "leg", "scalpel", "procedure", "disease", "sprain", "cured", "transfusion", "consulted", "dialysis", "drained"]
heatWords = filterLabels
data = zeros(length(heatWords),length(heatWords))

for i in 1:length(heatWords)
    println("")
    println(heatWords[i], propOccur(heatWords[i]))
end

for i in 1:length(heatWords)
    print(i/length(heatWords))
    for ii in 1:length(heatWords)
        if i == ii
            data[i,ii] = 1
        else
            data[i,ii] = propOccur(heatWords[i],heatWords[ii])/propOccur(heatWords[i])
        end
    end
end

heatmap(heatWords,
    heatWords, data, x_ticks = :all,xrotation = 45,y_ticks = :all,
    c=range(HSL(colorant"red"), stop=HSL(colorant"green"), length=15),
    title="Pr(Row|Column) in a Doctor's Transcript")

#Reorganizing Data for violin chart
blank = [[] for i = 1:length(unique(field))]
for i in 1:length(unique(field))
    for ii in 1:length(field)
        if field[ii] == unique(field)[i]
            push!(blank[i], length(split(trans[ii]," ")))
        end
    end
end

violin(reshape(unique(field),(1,29)), blank, leg = false, x_ticks = :all, xrotation = 60)
