using LinearAlgebra
using Distributions
using Plots

# -----------------------------------------------------------------------------
# Functions 
# -----------------------------------------------------------------------------

function gradient_descent( y, X, lr, logistic_regression, precision)

	# Initialize the algorithm 
	w = rand(p) .* 2*sqrt(6/p) .+ (-sqrt(6/p));
	neg_log_fn, gradient = logistic_regression( y, X, w);
	trace = [neg_log_fn + max( 1, 2*precision)];
	num_iter = 0;

	# Begin the gradient descent algorithm
	while abs(trace[end] - neg_log_fn) > precision 
		
		# Record the negative log-likelihood of the parameters
		push!( trace, neg_log_fn);
		
		# Update the parameters
		w = w .- lr .* gradient';
		
		# Compute negative log-likelihood and gradient for the updated parameters
		neg_log_fn, gradient = logistic_regression( y, X, w);
		
		# If negative log-likelihood increases, then reduce learning rate
		if trace[end] - neg_log_fn < 0
			lr = lr * .9;
		end
		
		num_iter +=1;
	end
	
	return([ w, trace, num_iter, lr])
end



function logistic_regression( y, X, w)
	
	# Compute the gradient at the parameters w, analytically
	temp = (1 ./ (1 .+exp.(-X * w))) .- y;
	gradient = sum(Diagonal(vec(temp)) * X, dims = 1);
	
	# Evaluate the negative log-likelihood at the parameters w
	sigmoid = (1 ./ (1 .+exp.(-X * w)));
	neg_log_fn = -sum(y .* log.(sigmoid) + (1 .- y) .* log.(1 .- sigmoid));
	
	return([ neg_log_fn, gradient])
end

# -----------------------------------------------------------------------------




n = 1000;
beta = [.5,1,2,3,4,5];
p = length(beta);

lr=.01;
precision = 10^(-30);


# Simulate "num_sims" data sets to sample from the sampling distribution of the
# coefficients
num_sims = 1000;
sampling_dist = zeros( num_sims, p);
for k in 1:num_sims
	
	# Generate a synthetic data set
	X = rand(Normal(0, 1), n, p);
	py = 1 ./ ( 1 .+ exp.(-X * beta));
	y = rand(n) .< py;

	output = gradient_descent( y, X, lr, logistic_regression, precision);
	sampling_dist[k,:] = output[1]
	println(k)
end

histogram( sampling_dist, layout=p, legend=false, 
           bins=floor(Int,sqrt(num_sims)),
           title=["true coef = "*"$i" for j in 1:1, i in beta])
plot!( ytickfontsize=Int(5), xtickfontsize=Int(5))
savefig("sampling_dist.pdf")

