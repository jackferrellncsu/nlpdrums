

xs = 5 .* rand(500,5)
ys = []
rule = [2,3,-5,2,-1]
for i in 1:500
    push!(ys, sum(xs[i,:] .* rule))
end

H = xs * inv(xs'*xs) * xs'

e_i = (H * ys) ./ (1 .- diag(H))

pval = []
for i in 1:length(e_i)
    push!(pval, sum(e_i .> e_i[i])/length(e_i))
end
