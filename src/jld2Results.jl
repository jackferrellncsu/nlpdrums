function jld2Results(errors, preds, trues)
    err = []
    for i in 1:1000
        if get(errors, "run"*string(i),0) != 0
            push!(err, get(errors, "run"*string(i),0))
            push!(preds, get(preds, "run"*string(i),0))
            push!(trues, get(trues, "run"*string(i),0))
        end
    end
    best = argmin(errors)
    worst = argmax(errors)
    return [err[best], preds[best], trues[best], err[worst], preds[worst], trues[best]]
end
