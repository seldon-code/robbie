#pragma once
#include "defines.hpp"
#include <optional>

namespace Robbie
{

template<typename scalar>
class Optimizer
{
public:
    Optimizer()          = default;
    virtual ~Optimizer() = default;
    using opt_matrix_t   = Matrix<scalar> *;
    using opt_vector_t   = Vector<scalar> *;

    virtual void optimize(
        opt_matrix_t matrix_variable, opt_matrix_t matrix_gradient, opt_vector_t vector_variable,
        opt_vector_t vector_gradient )
        = 0;
};
} // namespace Robbie