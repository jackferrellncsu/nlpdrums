using JLD
using MLBase
using Plots


struct ModelResult
    bestErrors
    bestPreds
    bestTrues
    worstErrors
    worstPreds
    worstTrues
    meanErr
end


medPreds = load("PredsFinal_ourdata.jld")
medError = load("ErrorsFinal_ourdata.jld")
medTrues = load("TruesFinal_ourdata.jld")

synthPreds = load("PredsFinal_synth.jld")
synthError = load("ErrorsFinal_synth.jld")
synthTrues = load("TruesFinal_synth.jld")

resultMed = jld2Results_j(medError, medPreds, medTrues)
resultSynth = jld2Results_j(synthError, synthPreds, synthTrues)

p = drawROC(resultMed, "Medical Data")

p2 = drawROC(resultSynth, "Synthetic Data")




function drawROC(R::ModelResult, studytype::String)
    bestBit = R.bestTrues .== 1.0
    bestRocs = roc(bestBit, vec(R.bestPreds))

    bTpr, bFpr = getPostiveRates(bestRocs)

    worstBit = R.worstTrues .== 1.0

    if R.worstPreds[1] == mean(R.worstPreds)
        R.worstPreds' .+= rand(length(R.worstPreds)) / 1000
    end
    worstRocs = roc(worstBit, vec(R.worstPreds))

    wTpr, wFpr = getPostiveRates(worstRocs)


    p = plot(bFpr, bTpr, label = "Best Model")
    plot!(p, wFpr, wTpr, label = "Worst Model")
    title!(p, studytype * " FF GloVe ROC Curve")
    xlabel!(p, "False Positive Rate")
    ylabel!(p, "True Positive Rate")

    return p
end


function getPostiveRates(rocNums)
    tpr = true_positive_rate.(rocNums)
    fpr = false_positive_rate.(rocNums)

    return tpr, fpr
end


function jld2Results_j(errors, preds, trues)
    err = []
    p = []
    t = []
    for i in 1:1000
        if get(errors, "run"*string(i),0) != 0
            push!(err, get(errors, "run"*string(i),0))
            push!(p, get(preds, "run"*string(i),0))
            push!(t, get(trues, "run"*string(i),0))
        end
    end

    worst = argmax(err)
    deleteat!(err, worst)
    deleteat!(p, worst)
    deleteat!(t, worst)

    best = argmin(err)
    worst = argmax(err)
    return ModelResult(err[best],
     p[best],
     t[best],
     err[worst],
     p[worst],
     t[worst],
     mean(err))
end
