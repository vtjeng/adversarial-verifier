include("nn_ops.jl")

batch = 1
in1_height = 8
in1_width = 8
stride1_height = 2
stride1_width = 2
pooled1_height = round(Int, in1_height/stride1_height, RoundUp)
pooled1_width = round(Int, in1_width/stride1_width, RoundUp)
in1_channels = 1
filter1_height = 4
filter1_width = 4
out1_channels = 4

in2_height = filter1_height
in2_width = filter1_width
stride2_height = 2
stride2_width = 2
pooled2_height = round(Int, in2_height/stride2_height, RoundUp)
pooled2_width = round(Int, in2_width/stride2_width, RoundUp)
in2_channels = out1_channels
filter2_height = 2
filter2_width = 2
out2_channels = 2
bigM = 10000

A_height = 4
A_width = pooled2_height*pooled2_width*out2_channels

B_height = 2
B_width = A_height

srand(5)
x_current = rand(-10:10, batch, in1_height, in1_width, in1_channels)
filter1 = rand(-10:10, filter1_height, filter1_width, in1_channels, out1_channels)
filter2 = rand(-10:10, filter2_height, filter2_width, in2_channels, out2_channels)
A = rand(-10:10, A_height, A_width)
B = rand(-10:10, B_height, B_width)

using JuMP
using Gurobi

m = Model(solver=GurobiSolver())

@variable(m, ve[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels])
@variable(m, vx[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels])
@constraint(m, vx .== x_current + ve) # input

vx1 = NNOps.convlayerconstraint(m, vx, filter1, (stride1_height, stride1_width), 10000)
vx2 = NNOps.convlayerconstraint(m, vx1, filter2, (stride2_height, stride2_width), 10000)
vx3 = NNOps.fullyconnectedlayerconstraint(m, vx2[:], A, 10000)
NNOps.softmaxconstraint(m, vx3, B, 1)

@objective(m, Min, sum(ve.^2))
status = solve(m)

println("Objective value: ", getobjectivevalue(m))
println("e = ", getvalue(ve))
