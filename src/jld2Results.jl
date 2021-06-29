using JLD
using ROCAnalysis
using MLBase
using Plots

function jld2Results(errors, preds, trues)
    err = []
    p = []
    t = []
    for i in 1:50
        if get(errors, "run"*string(i),0) != 0
            push!(err, get(errors, "run"*string(i),0))
            push!(p, get(preds, "run"*string(i),0))
            push!(t, get(trues, "run"*string(i),0))
        end
    end

    print(length(err))
    best = argmin(err)
    worst = argmax(err)
    print(worst)
    return [err[best],p[best],t[best],err[worst],p[worst], t[worst],mean(err)]
end

be,bp,bt,we,wp,wt,ae = jld2Results(JLD.load("ErrorsFinal.jld"),JLD.load("PredsFinal.jld"),JLD.load("TruesFinal.jld"))

rocBest = MLBase.roc(bt .== 1,convert(Vector{Float64},vec(bp')),1000)
rocWorst = MLBase.roc(wt .== 1,convert(Vector{Float64},vec(wp')),1000)

TP = []
FP = []
for i in 1:length(rocBest)
    push!(TP,rocBest[i].tp/rocBest[i].p)
    push!(FP,rocBest[i].fp/rocBest[i].n)
end
push!(TP,0.0)
push!(FP,0.0)

Plots.plot(FP,TP, label = "Best Model")

TP = []
FP = []
for i in 1:length(rocWorst)
    push!(TP,rocWorst[i].tp/rocWorst[i].p)
    push!(FP,rocWorst[i].fp/rocWorst[i].n)
end
push!(TP,0.0)
push!(FP,0.0)

Plots.plot!(FP,TP, label = "Worst Model")

xlabel!("False Positive Rate")
ylabel!("True Positive Rate")
Plots.title!("Varied Window Length Neural Net")

xaxis!([0,1])
yaxis!([0,1])
