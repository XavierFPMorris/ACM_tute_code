using Plots
using LinearAlgebra
plotly()

nsteps = 1000

h = 0.01

p = Array{Float64, 1}(undef, nsteps)
q = Array{Float64, 1}(undef, nsteps)


p[1] = 1
q[1] = 1

for i = 2:nsteps
    q[i] = q[i-1] + h*p[i-1]
    p[i] = p[i-1] - h*q[i-1]
end 

#plot(q,p, aspect_ratio=:equal)

### Implicit

p = Array{Float64, 1}(undef, nsteps)
q = Array{Float64, 1}(undef, nsteps)


p[1] = 1
q[1] = 1

A = [0 -1;1 0]

for i = 2:nsteps
    yn = [p[i-1], q[i-1]]
    yn1 = inv(I(2)-h*A)*yn
    p[i],q[i] = yn1[1], yn1[2]
end 

plot(q,p,aspect_ratio=:equal)