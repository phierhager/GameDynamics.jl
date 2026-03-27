module TabularTraits

export AbstractTabularModel
export AbstractTabularNormalFormModel
export AbstractTabularMarkovModel
export AbstractTabularExtensiveFormModel

export model_role
export is_solver_grade
export supports_exact_solvers
export supports_approx_solvers
export is_tree_model
export is_graph_model

abstract type AbstractTabularModel end
abstract type AbstractTabularNormalFormModel <: AbstractTabularModel end
abstract type AbstractTabularMarkovModel <: AbstractTabularModel end
abstract type AbstractTabularExtensiveFormModel <: AbstractTabularModel end

"""
Lightweight model-role classifier for diagnostics and error messages.
"""
model_role(::AbstractTabularModel) = :tabular_model
model_role(::AbstractTabularNormalFormModel) = :tabular_normal_form
model_role(::AbstractTabularMarkovModel) = :tabular_markov
model_role(::AbstractTabularExtensiveFormModel) = :tabular_extensive_form

"""
Whether the model is intended for solver use rather than diagnostics only.
"""
is_solver_grade(::AbstractTabularModel) = true

"""
Whether exact solvers should accept this model by default.
"""
supports_exact_solvers(::AbstractTabularModel) = true

"""
Whether approximate solvers should accept this model by default.
"""
supports_approx_solvers(::AbstractTabularModel) = false

"""
Tree/graph traits for extensive-form tabular models.
"""
is_tree_model(::AbstractTabularModel) = false
is_graph_model(::AbstractTabularModel) = false

end