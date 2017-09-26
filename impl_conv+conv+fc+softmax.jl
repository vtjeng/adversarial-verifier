using MAT
using JuMP
using Gurobi
include("nn_ops.jl")
include("util.jl")

### Parameters for neural net
batch = 1
in1_height = 14
in1_width = 14
stride1_height = 2
stride1_width = 2
pooled1_height = round(Int, in1_height/stride1_height, RoundUp)
pooled1_width = round(Int, in1_width/stride1_width, RoundUp)
in1_channels = 1
filter1_height = 3
filter1_width = 3
out1_channels = 4

in2_height = pooled1_height
in2_width = pooled1_width
stride2_height = 1
stride2_width = 1
pooled2_height = round(Int, in2_height/stride2_height, RoundUp)
pooled2_width = round(Int, in2_width/stride2_width, RoundUp)
in2_channels = out1_channels
filter2_height = 3
filter2_width = 3
out2_channels = 8

bigM = 10000

A_height = 64
A_width = pooled2_height*pooled2_width*out2_channels

B_height = 10
B_width = A_height

### Choosing data to be used
randomize = false
if randomize
    srand(5)
    x0 = rand(-10:10, batch, in1_height, in1_width, in1_channels)

    filter1 = rand(0:10, filter1_height, filter1_width, in1_channels, out1_channels)
    bias1 = rand(0:5, out1_channels)
    filter2 = rand(0:10, filter2_height, filter2_width, in2_channels, out2_channels)
    bias2 = rand(0:5, out2_channels)
    A = rand(-10:10, A_height, A_width)
    biasA = rand(-10:10, A_height)
    B = rand(-10:10, B_height, B_width)
    biasB = rand(-10:10, B_height)
else
    vars = matread("data/2017-09-26_144005_703000.mat")
    mnist_test_data_resized = matread("data/mnist_test_data_resized.mat")
    test_index = 1 # which test sample we're choosing
    y_ = mnist_test_data_resized["y_"]
    x_resize = mnist_test_data_resized["x_resize"]
    actual_label = get_label(y_, test_index)
    x0 = get_input(x_resize, test_index) # NB: weird indexing preserves singleton first dimension

    filter1 = vars["conv1/weight"]
    bias1 = squeeze(vars["conv1/bias"], 1)
    filter2 = vars["conv2/weight"]
    bias2 = squeeze(vars["conv2/bias"], 1)
    A = transpose(vars["fc1/weight"])
    biasA = squeeze(vars["fc1/bias"], 1)
    B = transpose(vars["fc2/weight"])
    biasB = squeeze(vars["fc2/bias"], 1)

    check_size(x0, (batch, in1_height, in1_width, in1_channels))
    check_size(filter1, (filter1_height, filter1_width, in1_channels, out1_channels))
    check_size(bias1, (out1_channels, ))
    check_size(filter2, (filter2_height, filter2_width, in2_channels, out2_channels))
    check_size(bias2, (out2_channels, ))
    check_size(A, (A_height, A_width))
    check_size(biasA, (A_height, ))
    check_size(B, (B_height, B_width))
    check_size(biasB, (B_height, ))

    function calculate_predicted_label{T<:Real}(
        x0_::AbstractArray{T, 4})::Int
        x1_ = NNOps.convlayer(x0_, filter1, bias1, (stride1_height, stride1_width))
        x2_ = NNOps.convlayer(x1_, filter2, bias2, (stride2_height, stride2_width))
        x3_ = NNOps.fullyconnectedlayer(permutedims(x2_, [4, 3, 2, 1])[:], A, biasA)
        predicted_label = NNOps.softmaxindex(x3_, B, biasB)
        return predicted_label
    end

    num_samples = 20
    num_correct = 0
    for i = 1:num_samples
        pred = calculate_predicted_label(get_input(x_resize, i))
        actual = get_label(y_, i)
        println("Running test case $i. Predicted is $pred, actual is $actual.")
        if pred == actual
            num_correct += 1
        end
    end
    println("Number correct is $num_correct out of $num_samples.")
end

## Calculate intermediate values
x1 = NNOps.convlayer(x0, filter1, bias1, (stride1_height, stride1_width))
x2 = NNOps.convlayer(x1, filter2, bias2, (stride2_height, stride2_width))
x3 = NNOps.fullyconnectedlayer(permutedims(x2, [4, 3, 2, 1])[:], A, biasA)
predicted_label = NNOps.softmaxindex(x3, B, biasB)

m = Model(solver=GurobiSolver())

@variable(m, ve[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels])
@variable(m, vx0[1:batch, 1:in1_height, 1:in1_width, 1:in1_channels])
@constraint(m, vx0 .== x0 + ve) # input

vx1 = NNOps.convlayerconstraint(m, vx0, filter1, bias1, (stride1_height, stride1_width), 10000)
vx2 = NNOps.convlayerconstraint(m, vx1, filter2, bias2, (stride2_height, stride2_width), 10000)
vx3 = NNOps.fullyconnectedlayerconstraint(m, permutedims(vx2, [4, 3, 2, 1])[:], A, biasA, 10000)
target_label = 3
setvalue(ve, get_input(x_resize, 2))
NNOps.softmaxconstraint(m, vx3, B, biasB, target_label)

@objective(m, Min, sum(ve.^2))

println("Attempting to find adversarial example. Neural net predicted label is $predicted_label, target label is $target_label")

solve(m)

println("Objective value: ", getobjectivevalue(m))
println("e = ", getvalue(ve))
