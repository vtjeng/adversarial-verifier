include("nn_ops.jl")

batch = 1
in_height = 6
in_width = 6
stride_height = 2
stride_width = 2
pooled_height = round(Int, in_height/stride_height, RoundUp)
pooled_width = round(Int, in_width/stride_width, RoundUp)
in_channels = 1
filter_height = 5
filter_width = 5
out_channels = 32
bigM = 10000

srand(5)
x_actual = rand(-10:10, batch, in_height, in_width, in_channels)
x_current = rand(-10:10, batch, in_height, in_width, in_channels)
filter = rand(-10:10, filter_height, filter_width, in_channels, out_channels)
x_conv_actual = NNOps.conv2d(x_actual, filter)
x_conv_relu_actual = NNOps.relu(x_conv_actual)
x_conv_relu_maxpool_actual = NNOps.maxpool(x_conv_relu_actual, (1, 2, 2, 1))

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
vx_conv = NNOps.conv2dconstraint(m, vx+ve, filter)
# @constraint(m, conv2d(vx+ve, filter) .== vx_conv)
# @constraint(m, vx_conv .== x_conv_actual)

# 2. Adding relu layer
vx_conv_relu = NNOps.reluconstraint(m, vx_conv, 10000)
# @constraint(m, vx_conv_relu .== x_conv_relu_actual)

# 3. Adding maxpool layer
vx_conv_relu_maxpool = NNOps.maxpoolconstraint(m, vx_conv_relu, (2, 2), 10000)
@constraint(m, vx_conv_relu_maxpool .== x_conv_relu_maxpool_actual)

@objective(m, Min, sum(ve.^2))

status = solve(m)

println("Objective value: ", getobjectivevalue(m))
println("e = ", getvalue(ve))
