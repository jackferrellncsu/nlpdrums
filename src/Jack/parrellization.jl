using CUDA
using Test


N = 2^20
x = fill(1.0f0, N)
y = fill(2.0f0, N)

Threads.nthreads()
y .+= x

function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
sequential_add!(y, x)
@test all(y.==3.0)

function parrallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
parrallel_add!(y, x)
@test all(y  .== 3.0f0)

using BenchmarkTools
@btime sequential_add!($y, $x)

@btime parrallel_add!($y, $x)

x_d = CUDA.fill(1.0f0, N)
y_d = CUDA.fill(2.0f0, N)

y_d .+= x_d
@test all(Array(y_d) .== 3.0f0)

function add_broadcast!(y, x)
    CUDA.@sync y .+= x
    return
end

@btime add_broadcast!($y_d, $x_d)
