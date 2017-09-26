include("nn_ops.jl")

batch = 1
in1_height = 14
in1_width = 14
stride1_height = 1
stride1_width = 1
pooled1_height = round(Int, in1_height/stride1_height, RoundUp)
pooled1_width = round(Int, in1_width/stride1_width, RoundUp)
in1_channels = 1
filter1_height = 3
filter1_width = 3
out1_channels = 4

in2_height = pooled1_height
in2_width = pooled1_width
stride2_height = 2
stride2_width = 2
pooled2_height = round(Int, in2_height/stride2_height, RoundUp)
pooled2_width = round(Int, in2_width/stride2_width, RoundUp)
in2_channels = out1_channels
filter2_height = 5
filter2_width = 5
out2_channels = 8

bigM = 10000

A_height = 64
A_width = pooled2_height*pooled2_width*out2_channels

B_height = 10
B_width = A_height

srand(5)
x0 = rand(-10:10, batch, in1_height, in1_width, in1_channels)
filter1 = rand(0:10, filter1_height, filter1_width, in1_channels, out1_channels)
filter2 = rand(0:10, filter2_height, filter2_width, in2_channels, out2_channels)
A = rand(-10:10, A_height, A_width)
B = rand(-10:10, B_height, B_width)

println(size(filter1))
println(size(filter2))
println(size(A))
println(size(B))

## Calculate intermediate values
x1 = NNOps.convlayer(x0, filter1, (stride1_height, stride1_width))
x2 = NNOps.convlayer(x1, filter2, (stride2_height, stride2_width))
x3 = NNOps.fullyconnectedlayer(x2[:], A)
println(B*x3)
target_index = NNOps.softmaxindex(x3, B)



using JuMP
using Gurobi

m = Model(solver=GurobiSolver())

@variable(m, ve[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels])
@variable(m, vx0[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels])
@constraint(m, vx0 .== x0 + ve) # input

vx1 = NNOps.convlayerconstraint(m, vx0, filter1, (stride1_height, stride1_width), 10000)
vx2 = NNOps.convlayerconstraint(m, vx1, filter2, (stride2_height, stride2_width), 10000)
vx3 = NNOps.fullyconnectedlayerconstraint(m, vx2[:], A, 10000)
NNOps.softmaxconstraint(m, vx3, B, 2)

@objective(m, Min, sum(ve.^2))
log_output = true;

solve(m)

println("Objective value: ", getobjectivevalue(m))
println("e = ", getvalue(ve))
