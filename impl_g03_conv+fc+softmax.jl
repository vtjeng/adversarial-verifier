using MAT
using JuMP
using Gurobi
include("nn_ops.jl")
include("util.jl")

"""
Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

...
H  749    58                       0.0347209    0.00000   100%  37.5    0s
H  781    50                       0.0170735    0.00000   100%  36.7    0s
H 1038   211                       0.0152771    0.00000   100%  37.8    0s
H 1479   245                       0.0128289    0.00000   100%  35.0    1s
* 1595   248              84       0.0115409    0.00000   100%  34.3    1s
H 2127   349                       0.0114459    0.00137  88.0%  32.3    1s
H 2128   316                       0.0097672    0.00137  86.0%  32.3    1s

Cutting planes:
  Learned: 1
  Cover: 4
  Implied bound: 9
  Clique: 2
  MIR: 108
  Flow cover: 69

Explored 16636 nodes (547268 simplex iterations) in 4.83 seconds
Thread count was 8 (of 8 available processors)

Solution count 10: 0.00976716 0.0114459 0.0115409 ... 0.255671
Pool objective bound 0.00976716

Optimal solution found (tolerance 1.00e-04)
Best objective 9.767161242031e-03, best bound 9.767161242031e-03, gap 0.0000%
Objective value: 0.00976716124203126
e = [0.0 0.0 0.0 0.00769844 -0.0219852 0.0 -0.00879821 0.0251259]
...
[0.0 0.0 0.0 -0.0100206 0.000587532 -0.010442 0.000612239 0.0]
"""

### Parameters for neural net
batch = 1
in1_height = 8
in1_width = 8
stride1_height = 2
stride1_width = 2
pooled1_height = round(Int, in1_height/stride1_height, RoundUp)
pooled1_width = round(Int, in1_width/stride1_width, RoundUp)
in1_channels = 1
filter1_height = 2
filter1_width = 2
out1_channels = 2

A_height = 5
A_width = pooled1_height*pooled1_width*out1_channels

B_height = 3
B_width = A_height

### Choosing data to be used
srand(5)
x0 = rand(batch, in1_height, in1_width, in1_channels)

filter1 = rand(filter1_height, filter1_width, in1_channels, out1_channels)*2-1
bias1 = rand(out1_channels)*2-1
A = rand(-10:10, A_height, A_width)
biasA = rand(-10:10, A_height)
B = rand(B_height, B_width)*2-1
biasB = rand(B_height)*2-1

## Calculate intermediate values
x1 = NNOps.convlayer(x0, filter1, bias1, (stride1_height, stride1_width))
x2 = NNOps.flatten(x1)
x3 = NNOps.fullyconnectedlayer(x2, A, biasA)
predicted_label = NNOps.softmaxindex(x3, B, biasB)

m = Model(solver=GurobiSolver(MIPFocus = 3))

@variable(m, ve[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels])
@variable(m, 0 <= vx0[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels] <= 1)
@constraint(m, vx0 .== x0 + ve) # input

vx1 = NNOps.convlayerconstraint(m, vx0, filter1, bias1, (stride1_height, stride1_width))
vx2 = NNOps.flatten(vx1)
vx3 = NNOps.fullyconnectedlayerconstraint(m, vx2, A, biasA)
target_label = 3
NNOps.softmaxconstraint(m, vx3, B, biasB, target_label, -1.0)

@objective(m, Min, sum(ve.^2))

println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

solve(m)

println("Objective value: ", getobjectivevalue(m))
println("e = ", getvalue(ve))
