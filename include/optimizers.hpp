#pragma once
#include "defines.hpp"
#include <fmt/ostream.h>
#include <cstddef>
#include <optional>
#include <vector>

namespace Robbie::Optimizers
{

template<typename scalar>
class Optimizer
{
protected:
    std::vector<Eigen::Ref<Robbie::Matrix<scalar>>> variables;
    std::vector<Eigen::Ref<Robbie::Matrix<scalar>>> gradients;

public:
    Optimizer() = default;

    virtual void clear()
    {
        variables.clear();
        gradients.clear();
    }

    virtual void register_variable( Eigen::Ref<Robbie::Matrix<scalar>> var, Eigen::Ref<Robbie::Matrix<scalar>> grad )
    {
        variables.push_back( var );
        gradients.push_back( grad );
    };

    virtual ~Optimizer()    = default;
    virtual void optimize() = 0;
};

template<typename scalar>
class DoNothing : public Optimizer<scalar>
{
public:
    DoNothing() = default;
    virtual void optimize(){};
};

template<typename scalar>
class StochasticGradientDescent : public Optimizer<scalar>
{

private:
    scalar learning_rate = 0.0;

public:
    StochasticGradientDescent( scalar learning_rate ) : learning_rate( learning_rate ) {}

    void optimize() override
    {
        for( size_t iv = 0; iv < this->variables.size(); iv++ )
        {
            this->variables[iv] -= learning_rate * this->gradients[iv];
        }
    }
};

template<typename scalar>
class Adam : public Optimizer<scalar>
{

    // private:
    //     scalar alpha   = 0.001;
    //     scalar beta1   = 0.9;
    //     scalar beta2   = 0.999;
    //     scalar epsilon = 1e-8;

    //     // first moments
    //     Matrix<scalar> m_matrix;
    //     Vector<scalar> m_vector;

    //     // second moments
    //     Matrix<scalar> v_matrix;
    //     Vector<scalar> v_vector;

    //     size_t timestep = 0;

    //     void initialize( Matrix<scalar> * matrix_variable, Vector<scalar> * vector_variable )
    //     {
    //         if( matrix_variable != nullptr )
    //         {
    //             m_matrix = Matrix<scalar>::Zero( matrix_variable->rows(), matrix_variable->cols() );
    //             v_matrix = Matrix<scalar>::Zero( matrix_variable->rows(), matrix_variable->cols() );
    //         }

    //         if( vector_variable != nullptr )
    //         {
    //             m_vector = Vector<scalar>::Zero( vector_variable->size() );
    //             v_vector = Vector<scalar>::Zero( vector_variable->size() );
    //         }
    //     }

    // public:
    //     Adam() = default;
    //     Adam( scalar alpha ) : alpha( alpha ) {}
    //     Adam( scalar alpha, scalar beta1, scalar beta2, scalar epsilon )
    //             : alpha( alpha ), beta1( beta1 ), beta2( beta2 ), epsilon( epsilon )
    //     {
    //     }

    //     void optimize(
    //         Matrix<scalar> * matrix_variable, Matrix<scalar> * matrix_gradient, Vector<scalar> * vector_variable,
    //         Vector<scalar> * vector_gradient ) override
    //     {
    //         if( timestep == 0 )
    //         {
    //             initialize( matrix_variable, vector_variable );
    //         }

    //         scalar beta_1_t = std::pow( beta1, timestep + 1 );
    //         scalar beta_2_t = std::pow( beta2, timestep + 1 );
    //         scalar alpha_t  = alpha * std::sqrt( 1.0 - beta_2_t ) / ( 1.0 - beta_1_t );

    //         if( matrix_variable != nullptr )
    //         {
    //             // Update first moments
    //             m_matrix = m_matrix * beta1 + ( 1.0 - beta1 ) * ( *matrix_gradient );
    //             // Update second moments
    //             v_matrix = v_matrix * beta2 + ( 1.0 - beta2 ) * matrix_gradient->array().pow( 2 ).matrix();
    //             *matrix_variable -= alpha_t * ( m_matrix.array() / ( v_matrix.array().sqrt() + epsilon ) ).matrix();
    //         }

    //         if( vector_variable != nullptr )
    //         {
    //             // Update first moments
    //             m_vector = m_vector * beta1 + ( 1.0 - beta1 ) * ( *vector_gradient );
    //             // Update second moments
    //             v_vector = v_vector * beta2 + ( 1.0 - beta2 ) * vector_gradient->array().pow( 2 ).matrix();
    //             *vector_variable -= alpha_t * ( m_vector.array() / ( v_vector.array().sqrt() + epsilon ) ).matrix();
    //         }

    //         timestep++;
    //     }
};

} // namespace Robbie::Optimizers