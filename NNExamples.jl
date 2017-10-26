if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

module NNExamples

using JuMP
using Gurobi

using NNOps
using NNParameters

function initialize{N}(
    nn_params::Union{NeuralNetParameters, LayerParameters},
    input_size::NTuple{N}; rebuild::Bool = false
    )::Tuple{JuMP.Model, Array{JuMP.Variable, N}, Array{JuMP.Variable, N}, Array}

    model_file_name = "models/$(nn_params.UUID).$(input_size).jls"
    if isfile(model_file_name) && !rebuild
        println("Loading model from cache.")
        open(model_file_name, "r") do f
            return deserialize(f)
        end
    else
        println("Rebuilding model from scratch.")
        r = initialize_uncached(nn_params, input_size)
        open(model_file_name, "w") do f
            serialize(f, r)
        end
        return r
    end
end

function initialize_uncached{N}(
    nn_params::Union{NeuralNetParameters, LayerParameters},
    input_size::NTuple{N}
    )::Tuple{JuMP.Model, Array{JuMP.Variable, N}, Array{JuMP.Variable, N}, Array}

    m = Model(solver=GurobiSolver(MIPFocus = 0, MIPGap=1.1))
    dummy = Array{Void}(input_size)

    v_input = map(_ -> @variable(m), dummy) # what you're trying to perturb
    v_e = map(_ -> @variable(m), dummy) # perturbation added
    v_x0 = map(_ -> @variable(m, lowerbound = 0, upperbound = 1), dummy) # perturbation + original image
    @constraint(m, v_x0 .== v_input + v_e)

    v_output = v_x0 |> nn_params

    setsolve(m, GurobiSolver(MIPFocus = 3))

    return (m, v_input, v_e, v_output)
end

end
