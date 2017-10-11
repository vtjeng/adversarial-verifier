if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

using MAT
using JuMP
using Gurobi

using NNExamples
using NNOps

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

filter1 = rand(filter1_height, filter1_width, in1_channels, out1_channels)*2-1
bias1 = rand(out1_channels)*2-1
filter2 = rand(filter2_height, filter2_width, in2_channels, out2_channels)*2-1
bias2 = rand(out2_channels)*2-1
A = rand(A_height, A_width)*2-1
biasA = rand(A_height)*2-1
B = rand(B_height, B_width)*2-1
biasB = rand(B_height)*2-1

NNExamples.solve_conv_conv_fc_softmax(x0, filter1, bias1, strides1, filter2, bias2, strides2, A, biasA, B, biasB, 2, -1.0, map(_ -> 0.0, x0))
