using JLD

function jld2Results(errors, preds, trues)
    err = []
    p = []
    t = []
    for i in 1:1000
        if get(errors, "run"*string(i),0) != 0
            push!(err, get(errors, "run"*string(i),0)[1])
            push!(p, get(preds, "run"*string(i),0)[1])
            push!(t, get(trues, "run"*string(i),0)[1])
        end
    end
    print(length(err))
    best = argmin(err)
    worst = argmax(err)
    return [err[best],
    p[best],
    t[best],
    err[worst],
    p[worst],
    t[worst],
    mean(err)]
end

eb,pb,tb,ew,pw,tw,ea = jld2Results(JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/ErrorsFinal.jld"),
            JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/PredsFinal.jld"),
            JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/TruesFinal.jld"))

# Printing an ROC curve for word2vec
pw .+= rand(length(pw))/1000000
best_p = convert(Vector{Float64}, pb)
worst_p = convert(Vector{Float64}, pw)
rocnums1 = MLBase.roc((tb .== 1), best_p)
rocnums2 = MLBase.roc((tw .== 1), worst_p)

emiTPR_worst = true_positive_rate.(rocnums2)
emiFPR_worst = false_positive_rate.(rocnums2)

emiTPR_best = true_positive_rate.(rocnums1)
emiFPR_best = false_positive_rate.(rocnums1)

Plots.plot(emiFPR_best, emiTPR_best, label = "Best Model")
Plots.plot!(emiFPR_worst,emiTPR_worst, label = "Worst Model")
Plots.title!("ROC Curve, Convolutional Layers with Synthetic Data")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")


x= rand(10)
y = x.*rand(10)

plot(x, y)
