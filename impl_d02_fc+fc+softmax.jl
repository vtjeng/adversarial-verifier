using MAT
using JuMP
using Gurobi
include("nn_ops.jl")
include("nn_examples.jl")
include("util.jl")

### Parameters for neural net
batch = 1
in1_height = 28
in1_width = 28

A_height = 40
A_width = in1_height*in1_width

B_height = 20
B_width = A_height

C_height = 10
C_width = B_height

UUID = "2017-10-09_201838"
nn_params = matread("data/$UUID-ch-params.mat")
x_reg = matread("data/$UUID-adversarial-examples.mat")["x"]
mnist_test_data_resized = matread("data/mnist_test_data_resized.mat")
test_index = 2 # which test sample we're choosing
y_ = mnist_test_data_resized["y_"]
x_resize = mnist_test_data_resized["x_resize"]
actual_label = get_label(y_, test_index)
x0 = get_input(x_reg, test_index) # NB: weird indexing preserves singleton first dimension

A = transpose(nn_params["fc1/weight"])
biasA = squeeze(nn_params["fc1/bias"], 1)
B = transpose(nn_params["fc2/weight"])
biasB = squeeze(nn_params["fc2/bias"], 1)
C = transpose(nn_params["logits/weight"])
biasC = squeeze(nn_params["logits/bias"], 1)

check_size(A, (A_height, A_width))
check_size(biasA, (A_height, ))
check_size(B, (B_height, B_width))
check_size(biasB, (B_height, ))
check_size(C, (C_height, C_width))
check_size(biasC, (C_height, ))

map(target_label -> NNExamples.solve_fc_fc_softmax(x0, A, biasA, B, biasB, C, biasC, target_label, 0.0, map(_ -> 0.0, x0)), 6:10)
