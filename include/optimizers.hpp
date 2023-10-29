#pragma once
#include "defines.hpp"
#include <fmt/ostream.h>
#include <optional>

namespace Robbie::Optimizers
{

template<typename scalar>
class Optimizer
{
public:
    Optimizer()          = default;
    virtual ~Optimizer() = default;

    virtual void optimize(
        Matrix<scalar> * matrix_variable, Matrix<scalar> * matrix_gradient, Vector<scalar> * vector_variable,
        Vector<scalar> * vector_gradient )
        = 0;
};

template<typename scalar>
class DoNothing : public Optimizer<scalar>
{
public:
    DoNothing() = default;

    virtual void optimize(
        [[maybe_unused]] Matrix<scalar> * matrix_variable, [[maybe_unused]] Matrix<scalar> * matrix_gradient,
        [[maybe_unused]] Vector<scalar> * vector_variable, [[maybe_unused]] Vector<scalar> * vector_gradient ){};
};

template<typename scalar>
class StochasticGradientDescent : public Optimizer<scalar>
{

private:
    scalar learning_rate = 0.0;

public:
    StochasticGradientDescent( scalar learning_rate ) : learning_rate( learning_rate ) {}

    void optimize(
        Matrix<scalar> * matrix_variable, Matrix<scalar> * matrix_gradient, Vector<scalar> * vector_variable,
        Vector<scalar> * vector_gradient ) override
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

template<typename scalar>
class Adam : public Optimizer<scalar>
{

private:
    scalar alpha   = 0.001;
    scalar beta1   = 0.9;
    scalar beta2   = 0.999;
    scalar epsilon = 1e-8;

    // first moments
    Matrix<scalar> m_matrix;
    Vector<scalar> m_vector;

    // second moments
    Matrix<scalar> v_matrix;
    Vector<scalar> v_vector;

    size_t timestep = 0;

    void initialize( Matrix<scalar> * matrix_variable, Vector<scalar> * vector_variable )
    {
        if( matrix_variable != nullptr )
        {
            m_matrix = Matrix<scalar>::Zero( matrix_variable->rows(), matrix_variable->cols() );
            v_matrix = Matrix<scalar>::Zero( matrix_variable->rows(), matrix_variable->cols() );
        }

        if( vector_variable != nullptr )
        {
            m_vector = Vector<scalar>::Zero( vector_variable->size() );
            v_vector = Vector<scalar>::Zero( vector_variable->size() );
        }
    }

public:
    Adam() = default;
    Adam( scalar alpha, scalar beta1, scalar beta2, scalar epsilon )
            : alpha( alpha ), beta1( beta1 ), beta2( beta2 ), epsilon( epsilon )
    {
    }

    void optimize(
        Matrix<scalar> * matrix_variable, Matrix<scalar> * matrix_gradient, Vector<scalar> * vector_variable,
        Vector<scalar> * vector_gradient ) override
    {
        if( timestep == 0 )
        {
            initialize( matrix_variable, vector_variable );
        }

        scalar beta_1_t = std::pow( beta1, timestep + 1 );
        scalar beta_2_t = std::pow( beta2, timestep + 1 );

        if( matrix_variable != nullptr )
        {
            // Update first moments
            m_matrix = m_matrix * beta1 + ( 1.0 - beta1 ) * ( *matrix_gradient );

            // Update second moments
            v_matrix = v_matrix * beta2 + ( 1.0 - beta2 ) * matrix_gradient->array().pow( 2 ).matrix();

            // Bias corrected first moments
            auto m_hat = m_matrix / ( 1.0 - beta_1_t );

            // Bias corrected second moments
            auto v_hat = v_matrix / ( 1.0 - beta_2_t );

            *matrix_variable -= alpha * ( m_hat.array() / ( v_hat.array().sqrt() + epsilon ) ).matrix();
        }

        if( vector_variable != nullptr )
        {
            // Update first moments
            m_vector = m_vector * beta1 + ( 1.0 - beta1 ) * ( *vector_gradient );

            // Update second moments
            v_vector = v_vector * beta2 + ( 1.0 - beta2 ) * vector_gradient->array().pow( 2 ).matrix();

            // Bias corrected first moments
            auto m_hat = m_vector / ( 1.0 - beta_1_t );

            // Bias corrected second moments
            auto v_hat = v_vector / ( 1.0 - beta_2_t );

            *vector_variable -= alpha * ( m_hat.array() / ( v_hat.array().sqrt() + epsilon ) ).matrix();
        }

        timestep++;
    }
};

} // namespace Robbie::Optimizers