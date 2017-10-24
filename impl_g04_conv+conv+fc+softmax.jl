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

in2_height = pooled1_height
in2_width = pooled1_width
stride2_height = 1
stride2_width = 1
strides2 = (1, stride2_height, stride2_width, 1)
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

conv1params = ConvolutionLayerParameters(
    rand(filter1_height, filter1_width, in1_channels, out1_channels)*2-1,
    rand(out1_channels)*2-1,
    strides1
)

conv2params = ConvolutionLayerParameters(
    rand(filter2_height, filter2_width, in2_channels, out2_channels)*2-1,
    rand(out2_channels)*2-1,
    strides2
)

fc1params = FullyConnectedLayerParameters(
    rand(A_height, A_width)*2-1,
    rand(A_height)*2-1
)

softmaxparams = SoftmaxParameters(
    rand(B_height, B_width)*2-1,
    rand(B_height)*2-1
)

nnparams = StandardNeuralNetParameters([conv1params, conv2params], [fc1params], softmaxparams)

(m, ve) = NNExamples.initialize(x0, nnparams, 2, -1.0)

abs_ve = NNOps.abs_ge.(ve)
e_norm = sum(abs_ve)
       
@objective(m, Min, e_norm)
       
status = solve(m)
