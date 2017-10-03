using MAT
using JuMP
using Gurobi
include("nn_ops.jl")
include("util.jl")

"""
pre- ConditionalJuMP

Attempting to find adversarial example. Neural net predicted label is 1, target label is 2
Optimize a model with 1150 rows, 818 columns and 3469 nonzeros
Model has 64 quadratic objective terms
Variable types: 525 continuous, 293 integer (293 binary)
...
Presolve removed 414 rows and 241 columns
Presolve time: 0.01s
Presolved: 736 rows, 577 columns, 3168 nonzeros
Presolved model has 64 quadratic objective terms
Variable types: 354 continuous, 223 integer (223 binary)
Presolve removed 93 rows and 93 columns
Presolved: 643 rows, 484 columns, 2889 nonzeros
Presolved model has 64 quadratic objective terms

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

...
     0     2    0.00000    0   98    7.43427    0.00000   100%     -    0s
...
 13061  9366    0.36796   51   89    0.58402    0.00101   100%  38.7   20s
...
124296 84436    0.08787   77   83    0.57558    0.08108  85.9%  47.9  201s

after ConditionalJuMP

ptimize a model with 2898 rows, 1010 columns and 6947 nonzeros
Model has 64 quadratic objective terms
Variable types: 525 continuous, 485 integer (485 binary)
Coefficient statistics:
  Matrix range     [2e-04, 2e+01]
...
Presolve removed 2044 rows and 496 columns
Presolve time: 0.01s
Presolved: 854 rows, 514 columns, 3362 nonzeros
Presolved model has 64 quadratic objective terms
Variable types: 313 continuous, 201 integer (201 binary)
Presolve removed 72 rows and 0 columns
Presolved: 782 rows, 586 columns, 3218 nonzeros
Presolved model has 64 quadratic objective terms

Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0    0.00000    0   97          -    0.00000      -     -    0s
...
H15584 10583                       0.5755818    0.07414  87.1%  34.2   15s
 17259 11921    0.19570   32  104    0.57558    0.07986  86.1%  36.8   20s
...
 134782 84413    0.38429   26   87    0.57558    0.32231  44.0%  51.2  200s
...
H672317 13446                       0.5755787    0.56984  1.00%  32.1  872s
...
 683248  3289     cutoff   40         0.57558    0.57405  0.27%  31.7  885s

...
Optimal solution found (tolerance 1.00e-04)
Best objective 5.755787176430e-01, best bound 5.755275600256e-01, gap 0.0089%
Objective value: 0.5755787176429682

e =
[0.0 0.0 0.0 -0.0508827 0.145311 0.0 -0.0192372 0.0549376]

[0.0 0.0 0.0 0.161038 -0.00944203 0.0203312 0.0028216 -0.00356975]

[0.0 0.0 0.0 -0.0672854 0.310163 -0.300806 -0.0600599 0.0371844]

[0.0 -0.0230471 0.0658179 0.249202 -0.28457 0.00380234 0.10872 -0.00806417]

[0.0 0.0226808 -0.00427672 0.0 0.0 -0.0334037 -0.087447 0.175153]

[0.0 -0.0311866 0.0890627 0.0 0.0413339 -0.118041 0.240889 -0.0165905]

[0.00201245 0.0929548 -0.00578713 -0.0466437 0.0023881 0.0304695 -0.0651105 0.0]

[-0.00636915 0.000373439 0.0 0.147622 -0.00865542 -0.0721574 0.00423076 0.0]
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

in2_height = pooled1_height
in2_width = pooled1_width
stride2_height = 1
stride2_width = 1
pooled2_height = round(Int, in2_height/stride2_height, RoundUp)
pooled2_width = round(Int, in2_width/stride2_width, RoundUp)
in2_channels = out1_channels
filter2_height = 2
filter2_width = 2
out2_channels = 4

A_height = 5
A_width = pooled2_height*pooled2_width*out2_channels

B_height = 3
B_width = A_height

### Choosing data to be used
srand(5)
x0 = rand(batch, in1_height, in1_width, in1_channels)

filter1 = rand(filter1_height, filter1_width, in1_channels, out1_channels)*2-1
bias1 = rand(out1_channels)*2-1
filter2 = rand(filter2_height, filter2_width, in2_channels, out2_channels)*2-1
bias2 = rand(out2_channels)*2-1
A = rand(A_height, A_width)*2-1
biasA = rand(A_height)*2-1
B = rand(B_height, B_width)*2-1
biasB = rand(B_height)*2-1

## Calculate intermediate values
x1 = NNOps.convlayer(x0, filter1, bias1, (stride1_height, stride1_width))
x2 = NNOps.convlayer(x1, filter2, bias2, (stride2_height, stride2_width))
x3 = NNOps.fullyconnectedlayer(NNOps.flatten(x2), A, biasA)
predicted_label = NNOps.softmaxindex(x3, B, biasB)

m = Model(solver=GurobiSolver(MIPFocus = 3))

@variable(m, ve[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels])
@variable(m, 0 <= vx0[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels] <= 1)
@constraint(m, vx0 .== x0 + ve) # input

vx1 = NNOps.convlayerconstraint(m, vx0, filter1, bias1, (stride1_height, stride1_width))
vx2 = NNOps.convlayerconstraint(m, vx1, filter2, bias2, (stride2_height, stride2_width))
vx3 = NNOps.fullyconnectedlayerconstraint(m, NNOps.flatten(vx2), A, biasA)
target_label = 2
NNOps.softmaxconstraint(m, vx3, B, biasB, target_label, -1.0)

@objective(m, Min, sum(ve.^2))

println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

solve(m)

println("Objective value: ", getobjectivevalue(m))
println("e = ", getvalue(ve))
