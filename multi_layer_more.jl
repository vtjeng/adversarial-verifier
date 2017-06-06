include("nn_ops.jl")

batch = 1
in_height = 4
in_width = 4
stride_height = 2
stride_width = 2
pooled_height = round(Int, in_height/stride_height, RoundUp)
pooled_width = round(Int, in_width/stride_width, RoundUp)
in_channels = 1
filter_height = 2
filter_width = 2
out_channels1 = 2
out_channels2 = 2
bigM = 10000

A_height = 10
A_width = pooled_height*pooled_width*out_channels2

B_height = 5
B_width = A_height

srand(5)
# x_actual = rand(-10:10, batch, in_height, in_width, in_channels)
x_current = rand(-10:10, batch, in_height, in_width, in_channels)
filter1 = rand(-10:10, filter_height, filter_width, in_channels, out_channels1)
filter2 = rand(-10:10, filter_height, filter_width, out_channels1, out_channels2)
A = rand(-10:10, A_height, A_width)
B = rand(-10:10, B_height, B_width)
# x1_actual = NNOps.convlayer(x_actual, filter1, (1, stride_height, stride_width, 1));
# x2_actual = NNOps.convlayer(x1_actual, filter2, (1, stride_height, stride_width, 1));
# x3_actual = NNOps.fullyconnectedlayer(x2_actual[:], A)

using JuMP
using Gurobi

m = Model(solver=GurobiSolver())

@variable(m, ve[1:batch, 1:in_height, 1:in_width, 1:in_channels])
@variable(m, vx[1:batch, 1:in_height, 1:in_width, 1:in_channels])
@variable(m, vx_conv[1:batch, 1:in_height, 1:in_width, 1:out_channels1])
@constraint(m, vx .== x_current) # input

vx1 = NNOps.convlayerconstraint(m, vx+ve, filter1, (stride_height, stride_width), 10000)
vx2 = NNOps.convlayerconstraint(m, vx1, filter2, (stride_height, stride_width), 10000)
vx12 = NNOps.fullyconnectedlayerconstraint(m, vx2[:], A, 10000)
NNOps.softmaxconstraint(m, vx12, B, 1)

@objective(m, Min, sum(ve.^2))
status = solve(m)

println("Objective value: ", getobjectivevalue(m))
# TODO: Are jump solutions global? Can I save particular variables?
println("e = ", getvalue(ve))


# x1_pert = NNOps.convlayer(x_current+getvalue(ve), filter1, (1, 2, 2, 1));
# x2_pert = NNOps.fullyconnectedlayer(x1_pert[:], A)
# softmax_pert = B*x2_pert
#
#
# x1_current = NNOps.convlayer(x_current, filter1, (1, 2, 2, 1));
# x2_current = NNOps.fullyconnectedlayer(x1_current[:], A)
# softmax_current = B*x2_current
