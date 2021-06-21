using Lathe.preprocess: TrainTestSplit
using DataFrames
using Flux
using GLM
using Plots
using LinearAlgebra
samples = 10000

function generateSimpleData(samples)
    dat = []
    class = []
    for i in 1:samples
        r = rand()
        push!(dat, r)
        push!(class, r > .4 + rand()./5)
    end
    return [dat,class]
end

dat, class = generateSimpleData(samples)
df = DataFrame(hcat(dat,class), :auto)
classifierData, calibrationData = TrainTestSplit(df, .9);
plot(dat,class, seriestype = :scatter)

function NN1(dfDat, predx, predy)
    data = dfDat[:,1]
    class = dfDat[:,2]
    sameclass = []
    diffclass = []
    for i in 1:length(class)
        if predy == class[i] && predx != data[i]
            push!(sameclass, norm(data[i] .- predx))
        elseif predx != data[i]
            push!(diffclass, norm(data[i] .- predx))
        end
    end
    return (minimum(sameclass)/minimum(diffclass))
end

function NN0(logit, predx, predy)
    if predy == 1
        return 1+exp(-coef(logit)'*[1,predx])
    else
        return 1+exp(coef(logit)'*[1,predx])
    end
end

function pValue(calibrationData,x, y)
    r = NN1(calibrationData,x,y)
    #r = NN0(logit,x,y)
    nonconf = []
    for i in 1:size(calibrationData)[1]
        push!(nonconf, NN1(calibrationData,calibrationData[i,1],calibrationData[i,2]))
        #push!(nonconf, NN0(logit,calibrationData[i,1],calibrationData[i,2]))
    end
    return sum(nonconf .> r)/length(nonconf)
end

logit = glm(@formula(x2 ~ x1),classifierData, Bernoulli(), LogitLink())

testor = .5
sig = .01
nums = []
rez = []
pValue(calibrationData,.6,0)
pValue(calibrationData,.6,1)
for i in 1:25
    println(i)
    if pValue(calibrationData,i/25,0) > sig
        push!(nums, i/25)
        push!(rez,0)
    end
    if pValue(calibrationData,i/25,1) > sig
        push!(nums, i/25)
        push!(rez,1)
    end
end

plot(nums,rez,seriestype = :scatter)
