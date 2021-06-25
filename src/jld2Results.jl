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
    t[best],
    mean(err)]
end
