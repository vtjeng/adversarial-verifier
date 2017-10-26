if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

using MAT
using JuMP
using Gurobi

using NNExamples
using NNOps
using NNParameters

### Parameters for neural net
batch = 1
in1_height = 8
in1_width = 8
stride1_height = 2
stride1_width = 2
strides1 = (1, stride1_height, stride1_width, 1)
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

srand(5)
input_size = (batch, in1_height, in1_width, in1_channels)
x0 = rand(input_size)

conv1params = ConvolutionLayerParameters(
    rand(filter1_height, filter1_width, in1_channels, out1_channels)*2-1,
    rand(out1_channels)*2-1,
    strides1
)

fc1params = FullyConnectedLayerParameters(
    rand(-10:10, A_height, A_width),
    rand(-10:10, A_height)
)

softmaxparams = SoftmaxParameters(
    rand(B_height, B_width)*2-1,
    rand(B_height)*2-1
)

nnparams = StandardNeuralNetParameters(
    [conv1params], 
    [fc1params], 
    softmaxparams,
    "g03"
)

(m, v_input, v_e, v_output) = NNExamples.initialize(nnparams, input_size)

# Set perturbation constraint
abs_v_e = NNOps.abs_ge.(v_e)
e_norm = sum(abs_v_e)
@objective(m, Min, e_norm)

# Set input constraint
NNOps.set_input_constraint(v_input, x0)

target_label = 3
NNOps.set_max_index(v_output, target_label, 1.0)
println("Attempting to find adversarial example. Neural net predicted label is $(x0 |> nnparams |> NNOps.get_max_index), target label is $target_label")
status = solve(m)
