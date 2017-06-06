include("nn_ops.jl")

batch = 1
in_height = 10
in_width = 10
stride_height = 2
stride_width = 2
pooled_height = round(Int, in_height/stride_height, RoundUp)
pooled_width = round(Int, in_width/stride_width, RoundUp)
in_channels = 1
filter_height = 2
filter_width = 2
out_channels = 2
bigM = 10000

A_height = 10
A_width = pooled_height*pooled_width*out_channels

B_height = 5
B_width = A_height

srand(5)
x_actual = rand(-10:10, batch, in_height, in_width, in_channels)
x_current = rand(-10:10, batch, in_height, in_width, in_channels)
filter = rand(-10:10, filter_height, filter_width, in_channels, out_channels)
A = rand(-10:10, A_height, A_width)
B = rand(-10:10, B_height, B_width)
x1_actual = NNOps.convlayer(x_actual, filter, (1, stride_height, stride_width, 1));
x2_actual = NNOps.fullyconnectedlayer(x1_actual[:], A)
# x_conv_actual = NNOps.conv2d(x_actual, filter)
# x_conv_relu_actual = NNOps.relu(x_conv_actual)
# x_conv_relu_maxpool_actual = NNOps.maxpool(x_conv_relu_actual, (1, 2, 2, 1));

using JuMP
using Gurobi

m = Model(solver=GurobiSolver())

@variable(m, ve[1:batch, 1:in_height, 1:in_width, 1:in_channels])
@variable(m, vx[1:batch, 1:in_height, 1:in_width, 1:in_channels])
@variable(m, vx_conv[1:batch, 1:in_height, 1:in_width, 1:out_channels])
# @variable(m, vx_conv_relu[1:batch, 1:in_height, 1:in_width, 1:out_channels])
# @variable(m, vx_conv_relu_maxpool[1:batch, 1:pooled_height, 1:pooled_width, 1:out_channels])
@constraint(m, vx .== x_current) # input

# 1. Only using convolution constraint
# vx_conv = NNOps.conv2dconstraint(m, vx+ve, filter)
# @constraint(m, conv2d(vx+ve, filter) .== vx_conv)
# @constraint(m, vx_conv .== x_conv_actual)

# 2. Adding relu layer
# vx_conv_relu = NNOps.reluconstraint(m, vx_conv, 10000)
# @constraint(m, vx_conv_relu .== x_conv_relu_actual)

# 3. Adding maxpool layer
# vx_conv_relu_maxpool = NNOps.maxpoolconstraint(m, vx_conv_relu, (2, 2), 10000)
# @constraint(m, vx_conv_relu_maxpool .== x_conv_relu_maxpool_actual)

# 1-3a. Directly adding constraints
vx1 = NNOps.convlayerconstraint(m, vx+ve, filter, (stride_height, stride_width), 10000)
# @constraint(m, vx1 .== x1_actual)

# 4. Add fully connected layer
vx2 = NNOps.fullyconnectedlayerconstraint(m, vx1[:], A, 10000)
# @constraint(m, vx2 .== x2_actual)

# 5. add softmax layer
NNOps.softmaxconstraint(m, vx2, B, 1)

@objective(m, Min, sum(ve.^2))

status = solve(m)

println("Objective value: ", getobjectivevalue(m))
# TODO: Are jump solutions global? Can I save particular variables?
println("e = ", getvalue(ve))


x1_pert = NNOps.convlayer(x_current+getvalue(ve), filter, (1, 2, 2, 1));
x2_pert = NNOps.fullyconnectedlayer(x1_pert[:], A)
softmax_pert = B*x2_pert


x1_current = NNOps.convlayer(x_current, filter, (1, 2, 2, 1));
x2_current = NNOps.fullyconnectedlayer(x1_current[:], A)
softmax_current = B*x2_current
