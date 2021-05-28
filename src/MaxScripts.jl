using CSV
using DataFrames
using Plots
using StatsBase
using Languages

frame = CSV.read("/Users/mlovig/Documents/GitHub/nlpdrums/src/cleanedData.csv", DataFrame)

function Sampler(trans,field,spec)
    nField = []
    nTrans = []
    indexs = []

    for i in 1:length(trans)
        if field[i] == spec
            push!(nTrans,trans[i])
            push!(nField, field[i])
        end
    end

    for i in 1:length(trans)
        if field != spec
            if rand() <= length(nTrans)/(length(trans)-length(nTrans))
            push!(nTrans,trans[i])
            push!(nField,field[i])
            end
        end
    end
    #=
    s = StatsBase.sample(indexs,n=length(nTrans); replace = false)

    for i in s
        push!(nField, field[i])
        push!(nTrans,trans[i])
    end
    =#
    return nField,nTrans
end

function createTDM(frame, specialty)
    trans = frame[:,3]
    field = frame[:,1]

    trans,field = Sampler(trans,field,specialty)

    labels = []
    for x in trans
        labels = append!(labels,(split(x, " ")))
    end

    labels = unique(labels)
    filterLabels = []

    for i in 1:length(labels)
        if labels[i] ∉ stopwords(Languages.English()) #&& count(j->j==labels[i], split(trans[i])))>minOccur
            push!(filterLabels, labels[i])
        end
    end


    TDMatrix = [[] for i = 1:length(trans)+1]

    for i in 0:length(trans)
        println(i/length(trans))
        if i != 0
            push!(TDMatrix[i+1],field[i] == specialty)
            #push!(TDMatrix[i+1],trans[i])
            sp = split(trans[i], " ")
            println(count(j->j=="the",sp))
            for ii in filterLabels
                push!(TDMatrix[i+1],count(j->j==ii,sp))
            end
        else
            append!(TDMatrix[1], ["Label"])
            append!(TDMatrix[1], filterLabels)
        end
    end

    TDMFin = Array{Any}(undef, length(TDMatrix), length(TDMatrix[1]))

    for i in 1:size(TDMFin)[1]
        for ii in 1:size(TDMFin)[2]
            TDMFin[i,ii] = TDMatrix[i][ii]
        end
    end
    CSV.write("TermDocumentMatrix.csv",DataFrame(TDMFin[2:end,:],string.(TDMatrix[1])),writeheader = true)
    return TDMFin

end



F = createTDM(frame[1:end,:], " Cardiovascular / Pulmonary")



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
#=
low = 50
up = 2000
filterLabels = []
for i in 1:length(labels)
    if up >=counts[i] >=low
        push!(filterLabels,labels[i])
    end
end
=#

using Languages

filterLabels = []

for i in 1:length(labels)
    if labels[i] ∉ stopwords(Languages.English()) && counts[i]>10
        push!(filterLabels, labels[i])
    end
end

TDMatrix = [[] for i = 1:length(trans)+1]

for i in 0:length(trans)
    println(i/length(trans))
    if i != 0
        push!(TDMatrix[i+1],field[i])
        push!(TDMatrix[i+1],trans[i])
        for ii in 1:length(filterLabels)
            push!(TDMatrix[i+1],count(j->j==filterLabels[ii], split(trans[i])))
        end
    else
        append!(TDMatrix[1], ["Label", "Transcript"])
        append!(TDMatrix[1], filterLabels)
    end
end

Copy = copy(TDMatrix)

for i in 1:length(filterLabels)
    push!(TDMatrix[length(TDMatrix)],count(j->j==filterLabels[i], split(trans[length(TDMatrix)-1])))
end

TDMFin = Array{Any}(undef, length(TDMatrix), length(TDMatrix[1]))
for i in 1:size(TDMFin)[1]
    for ii in 1:size(TDMFin)[2]
        TDMFin[i,ii] = TDMatrix[i][ii]
    end
end

 CSV.write("TermDocumentMatrix.csv",DataFrame(TDMFin[2:end,:],string.(TDMatrix[1])),writeheader = true)

#=
Matching Transcripts with the fields
filterTrans = []
filterField = []
match = " Neurology"
for i in 1:length(trans)
    if field[i] == match
        push!(filterTrans,trans[i])
        push!(filterField, 1)
    end
end
=#

#Selecting without replacement from non-true sources
#=
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
    println(i/size(bigM)[1])
    for ii in 1:size(bigM)[2]
        bigM[i,ii] = count(j->j==filterLabels[ii], split(filterTrans[i]))
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
=#

filename1 = "TermDocumentMatrix.csv"
filepath1 = joinpath(@__DIR__, filename)
arr1 = CSV.read("/Users/mlovig/TermDocumentMatrix.csv", DataFrame)
