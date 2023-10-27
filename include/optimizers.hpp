#pragma once
#include "defines.hpp"
#include <optional>

namespace Robbie::Optimizers
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

template<typename scalar>
class DoNothing : public Optimizer<scalar>
{
public:
    DoNothing()        = default;
    using opt_matrix_t = typename Optimizer<scalar>::opt_matrix_t;
    using opt_vector_t = typename Optimizer<scalar>::opt_vector_t;

    virtual void optimize(
        [[maybe_unused]] opt_matrix_t matrix_variable, [[maybe_unused]] opt_matrix_t matrix_gradient,
        [[maybe_unused]] opt_vector_t vector_variable, [[maybe_unused]] opt_vector_t vector_gradient ){};
};

template<typename scalar>
class StochasticGradientDescent : public Optimizer<scalar>
{
    using opt_matrix_t = typename Optimizer<scalar>::opt_matrix_t;
    using opt_vector_t = typename Optimizer<scalar>::opt_vector_t;

private:
    scalar learning_rate = 0.0;

public:
    StochasticGradientDescent( scalar learning_rate ) : learning_rate( learning_rate ) {}

    void optimize(
        opt_matrix_t matrix_variable, opt_matrix_t matrix_gradient, opt_vector_t vector_variable,
        opt_vector_t vector_gradient ) override
    {
        if( matrix_variable != nullptr )
        {
            *( matrix_variable ) -= learning_rate * ( *matrix_gradient );
        }

        if( vector_variable != nullptr )
        {
            *( vector_variable ) -= learning_rate * ( *vector_gradient );
        }
    }
};

} // namespace Robbie::Optimizers