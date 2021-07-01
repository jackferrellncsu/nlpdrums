using JLD
using Statistics
using MLBase
using Plots
using LinearAlgebra

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
eb,pb,tb,ew,pw,tw,ea = jld2Results(JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/cluster_jld/ErrorsFinalREAL_.jld"),
            JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/cluster_jld/PredsFinalREAL_.jld"),
            JLD.load("/Users/eplanch/Documents/GitHub/nlpdrums/src/emi/cluster_jld/TruesFinalREAL_.jld"))

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

# CROSS VALIDATION ANALYSIS #

raw_matrix = JLD.load("/Users/mlovig/Documents/GitHub/nlpdrums/src/emi/cv_error_matrix.jld")
error_matrix = get(raw_matrix, "error_mat", 0)
error_matrix = error_matrix * 100

# Gets minimum value
min_vals = []
for i in 1:length(error_matrix[1, :])
    col_vec = error_matrix[:, i]
    min_val = minimum(col_vec)
    push!(min_vals, min_val)
end
argmin(min_vals)

# Gets average values
avg_vals = []
for i in 1:length(error_matrix[1, :])
    col_vec = error_matrix[:, i]
    sum_col = sum(col_vec)
    avg_col = sum_col / 100
    push!(avg_vals, avg_col)
end
argmin(avg_vals)


data = hcat(avg_vals[1:6], avg_vals[7:12], avg_vals[13:18], avg_vals[19:24], avg_vals[25:30], avg_vals[31:36])
heatmap([5,10,15,20,25,30],
    [200,300,400,500,600,700], data,
    c=:algae,
    xlabel="Convolution Filter Size", ylabel="Pooling Filter Size", title = "Cross-Validation for CNN",
    colorbar_title = "Percent Error")

x = 1:36
y = avg_vals
Plots.plot(x,y, title = "Average Error for Combinations", lw = 2, label = "Average Value")

x = 1:36
y = min_vals
Plots.plot!(x,y, title = "Minimum Error for Combinations", lw = 2, label = "Minimum Value")



Pkg.update()
ENV["GRDIR"]=""
Pkg.build("GR")
