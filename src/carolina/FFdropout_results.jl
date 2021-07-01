using JLD

errors = JLD.load("ErrorsFinal_cv.jld", "error_mat")
x = mapslices(mean, errors, dims = 1)
println(x)




best = argmin(errors)
worst = argmax(errors)

bestp = predictions[best]
worstp = predictions[worst]
bestt = trueValues[best]
worstt = trueValues[worst]
averageerr = mean(errors)

arbestp = convert(Vector{Float64}, bestp)
arworstp = convert(Vector{Float64}, worstp)
rocnumsbest = MLBase.roc(bestt.==1, arbestp, 50)
rocnumsworst = MLBase.roc(worstt.==1, arworstp, 50)

bestTPR = true_positive_rate.(rocnumsbest)
bestFPR = false_positive_rate.(rocnumsbest)
Plots.plot(bestFPR, bestTPR)

worstTPR = true_positive_rate.(rocnumsworst)
worstFPR = false_positive_rate.(rocnumsworst)
Plots.plot(bestFPR, bestTPR, label = "Best")
Plots.plot!(worstFPR, worstTPR, label = "Worst")
Plots.title!("ROC Curve, Dropout with Synthetic Data")
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")
