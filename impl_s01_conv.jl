include("nn_ops.jl")

# We specify the parameters for the size of the problem that we are solving.
batch = 1
in_height = 4
in_width = 4
stride_height = 2
stride_width = 2
in_channels = 1
filter_height = 2
filter_width = 2
out_channels = 2

pooled_height = round(Int, in_height/stride_height, RoundUp)
pooled_width = round(Int, in_width/stride_width, RoundUp)

k_in = 0.001

# Fix random seed so that we get consistent results
srand(5)

# We select a random filter for the convolution.
filter = rand(filter_height, filter_width, in_channels, out_channels)*2-1
bias = (rand(out_channels)*2-1)*0.25

using JuMP
using Gurobi

m = Model(solver=GurobiSolver(MIPFocus = 3))

# `ve` is the tensor of perturbations, while `vx` is the tensor representing the
# original input which we are perturbing.
# Our objective in this optimization is to minimize the l-2 norm of the
# perturbations.
@variable(m, ve[1:batch, 1:in_height, 1:in_width, 1:in_channels])
ve_abs = NNOps.abs_ge(m, ve)
@constraint(m, sum(ve_abs) <= k_in)
# @objective(m, Max, sum(ve.^2))

χ0 = rand(batch, in_height, in_width, in_channels)
@variable(m, 0 <= vx0[1:batch, 1:in_height, 1:in_width, 1:in_channels] <= 1)
@constraint(m, vx0 .== χ0 + ve)

χ1 = NNOps.convlayer(χ0, filter, bias, (2, 2))
vx1 = NNOps.convlayerconstraint(m, vx0, filter, bias, (2, 2))
dvx1_abs = NNOps.abs_le(m, vx1-χ1)
# @constraint(m, sum((vx1-x1)[:].^2) <= k_out)

@objective(m, Max, sum(dvx1_abs))

status = solve(m)

println("Objective value: ", getobjectivevalue(m))
println("\ne = ", getvalue(ve))
println("\nInput perturbation L1-norm is ", sum(abs.(getvalue(ve))))
x1 = NNOps.convlayer(χ0+getvalue(ve), filter, bias, (2, 2))
println("Output perturbation L1-norm (which should match objective) is actually ", sum(abs(x1 - χ1)))

println("\ndvx1_abs = ", getvalue(dvx1_abs))
println("\ndx1_abs = ", abs(x1 - χ1))

# println("\nvx1 = ", getvalue(vx1))
# println("\nx1 = ", x1)
# println("\nχ1 = ", χ1)
