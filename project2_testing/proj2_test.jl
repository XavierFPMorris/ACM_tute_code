using Plots
using LinearAlgebra

plotly()

Δt = 10
time = 200000
nsteps = time÷Δt

Q = zeros(Float64, (6,3,nsteps))
P = zeros(Float64, (6,3,nsteps))

## initial positions
q = [0 0 0
-3.5023653 -3.8169847 -1.5507963
9.0755314 -3.0458353 -1.6483708
8.3101420 -16.2901086 -7.2521278
11.4707666 -25.7294829 -10.8169456
-15.5387357 -25.2225594 -3.1902382]

## initial velocity
v = [0 0 0
0.00565429 -0.00412490 -0.00190589
0.00168318 0.00483525 0.00192462
0.00354178 0.00137102 0.00055029
0.00288930 0.00114527 0.00039677
0.00276725 -0.00170702 -0.00136504]

## mass
m =[1.00000597682
0.000954786104043
0.000285583733151
0.0000437273164546
0.0000517759138449
1/(1.3*10^8)]

## initial momentum
p = m.*v

Q[:,:,1] = q
P[:,:,1] = p

#print(Q)
#print(m[1])

#Gravity constant
G = 2.95912208286*10^-4

function ∂ₚH(i,t)
    P[i,:,t]/m[i]
end

function ∂ᵩH(i,t)
    temp = zeros(Float64, (3,))
    for j = 1:6
        if j != i
            temp += -G*m[i]*m[j]*(Q[j,:,t] - Q[i,:,t])./((norm(Q[i,:,t]-Q[j,:,t]))^3)
        end
    end
    return temp
end
sym = true
if sym
    for t = 2:nsteps
        for i = 1:6
            P[i,:,t] = P[i,:,t-1] - Δt*∂ᵩH(i,t-1)
            Q[i,:,t] = Q[i,:,t-1] + Δt*∂ₚH(i,t)
        end
    end
end
function split(A,i)
    B =A[i,1,:],A[i,2,:],A[i,3,:]
end

plot(split(Q,1), title = "Symplectic Euler Solar System")
plot!(xlabel = 'x', ylabel = 'y', zlabel = 'z')
#plot3d!(ztickfont = font(10, "sf"),ytickfont = font(10, "sf"),xtickfont = font(10, "sf"))

plot!(split(Q,2))
plot!(split(Q,3))
plot!(split(Q,4))
plot!(split(Q,5))
plot!(split(Q,6))
