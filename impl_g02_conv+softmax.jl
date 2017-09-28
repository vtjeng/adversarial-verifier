using MAT
using JuMP
using Gurobi
include("nn_ops.jl")
include("util.jl")

"""
Run results, caa 28-Sep-2017

     Nodes    |    Current Node    |     Objective Bounds      |     Work
  Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time
...
552363 113038     cutoff   43         0.22251    0.18083  18.7%  18.0  165s
...
917799  6622    0.22121   47   33    0.22251    0.22104  0.66%  18.3  305s

Cutting planes:
  Learned: 3
  Cover: 15
  Implied bound: 19
  MIR: 452
  Flow cover: 274

Explored 929515 nodes (16946499 simplex iterations) in 309.56 seconds
Thread count was 8 (of 8 available processors)

Solution count 10: 0.222512 0.2365 0.383976 ... 19.1794
Pool objective bound 0.222512

Optimal solution found (tolerance 1.00e-04)
Best objective 2.225117721602e-01, best bound 2.225117721602e-01, gap 0.0000%
Objective value: 0.22251177216018173

e = [0.0 0.0 -0.047361 0.118651 0.0 0.0 0.0128639 -0.0367368]
...
[-0.0301567 0.00176816 0.0 0.0 0.0 0.126407 -0.00741156 0.0]
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

B_height = 3
B_width = pooled1_height*pooled1_width*out1_channels

### Choosing data to be used
srand(5)
x0 = rand(batch, in1_height, in1_width, in1_channels)

filter1 = rand(filter1_height, filter1_width, in1_channels, out1_channels)*2-1
bias1 = rand(out1_channels)*2-1
B = rand(B_height, B_width)*2-1
biasB = rand(B_height)*2-1

## Calculate intermediate values
x1 = NNOps.convlayer(x0, filter1, bias1, (stride1_height, stride1_width))
x2 = NNOps.flatten(x1)
predicted_label = NNOps.softmaxindex(x2, B, biasB)

m = Model(solver=GurobiSolver(MIPFocus = 3))

@variable(m, ve[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels])
@variable(m, 0 <= vx0[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels] <= 1)
@constraint(m, vx0 .== x0 + ve) # input

vx1 = NNOps.convlayerconstraint(m, vx0, filter1, bias1, (stride1_height, stride1_width), 10000)
vx2 = NNOps.flatten(vx1)

target_label = 3
NNOps.softmaxconstraint(m, vx2, B, biasB, target_label, -1.0)

@objective(m, Min, sum(ve.^2))

println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

solve(m)

println("Objective value: ", getobjectivevalue(m))
println("e = ", getvalue(ve))
