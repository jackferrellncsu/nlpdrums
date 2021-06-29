using StatsBase
using DataFrames
using CSV
using Random
include("../data_cleaning.jl")

field = " Cardiovascular / Pulmonary"
data = importClean()
str = ""
strTrue = ""
strFalse = ""
for i in 1:size(data)[1]
    str = str * " " * data[i,3]
    if data[i,1] == field
        strTrue = strTrue* " " * data[i,3]
    else
        strFalse = strFalse* " " * data[i,3]
    end
end

splitStr = split(str, " ")
c = countmap(splitStr)
splitStrT = split(strTrue, " ")
cT = countmap(splitStrT)
splitStrF = split(strFalse, " ")
cF = countmap(splitStrF)

vals = vec(hcat([[val] for (key, val) in c]...))
words = vec(hcat([[key] for (key, val) in c]...))

valsTrue = zeros(length(words))
valsFalse = zeros(length(words))

for i in 1:length(words)
    valsTrue[i] = get(cT, words[i], 0)
    valsFalse[i] = get(cF, words[i], 0)
end

ST = sum(valsTrue)
SF = sum(valsFalse)

valsTrue = valsTrue ./ ST
valsFalse = valsFalse ./ SF
valsTrueCDF = zeros(length(valsTrue))
for i in 1:length(valsTrue)
        valsTrueCDF[i] = sum(valsTrue[1:i])
end
valsFalseCDF = zeros(length(valsFalse))
for i in 1:length(valsFalse)
        valsFalseCDF[i] = sum(valsFalse[1:i])
end

len = 100
mat = []
class = []
for i in 1:1000
    println(i)
    for ii in 1:500
        push!(mat,multinomialData(valsTrueCDF,len, words))
        push!(class,1)
    end
    for ii in 1:500
        push!(mat,multinomialData(valsFalseCDF,len, words))
        push!(class,0)
    end
end

df = DataFrame(hcat(mat,class))

CSV.write("JonsData.csv", df)

function multinomialData(probaCDF,len, words)
    trans = ""
    for i in 1:len
        trans = trans * words[getVal(probaCDF,rand())] * " "
    end
    return trans
end

function getVal(vec,num)
    for i in 1:length(vec)-1
        if num < vec[1]
            return 1
        end
        if i > 1 && num > vec[i-1] && num < vec[i]
            return i
        end
    end
    return length(vec)
end
