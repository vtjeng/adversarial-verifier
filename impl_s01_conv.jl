if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

using NNOps
using NNParameters

# We specify the parameters for the size of the problem that we are solving.
batch = 1
in_height = 4
in_width = 4
in_channels = 1
filter_height = 2
filter_width = 2
out_channels = 2

# Fix random seed so that we get consistent results
srand(5)

convparams = NNParameters.ConvolutionLayerParameters(
    rand(filter_height, filter_width, in_channels, out_channels)*2-1,
    (rand(out_channels)*2-1)*0.25,
    (1, 2, 2, 1)
)

input = rand(batch, in_height, in_width, in_channels)
lowerbounds = map(_ -> 0, input)
upperbounds = map(_ -> 1, input)

k_out = NNOps.convlayer_forwardprop(input, lowerbounds, upperbounds, convparams, 0.001)
println(k_out)
