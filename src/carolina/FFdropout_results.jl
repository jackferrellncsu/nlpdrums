using JLD

errors = JLD.load("ErrorsFinal_cv.jld", "error_mat")
avg_errors = mapslices(mean, errors, dims = 1)

# best = 70% Dropout, 1.64% error rate

x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Plots.plot(x, vec(avg_errors))
xlabel!("Dropout Rate")
ylabel!("Average Error Rate")
title!("Dropout Rate Cross Validation")
