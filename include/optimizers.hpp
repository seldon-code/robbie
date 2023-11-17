#pragma once
#include "defines.hpp"
#include <fmt/ostream.h>
#include <cstddef>
#include <optional>
#include <stdexcept>
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
        if( !( ( var.rows() == grad.rows() ) && ( var.cols() == grad.cols() ) ) )
        {
            throw std::runtime_error( "Tried to use variable and gradient of different shapes!" );
        }
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

private:
    scalar alpha   = 0.001;
    scalar beta1   = 0.9;
    scalar beta2   = 0.999;
    scalar epsilon = 1e-8;

    // first moments
    std::vector<Matrix<scalar>> m_matrix;
    // second moments
    std::vector<Matrix<scalar>> v_matrix;
    size_t timestep = 0;

public:
    void register_variable( Eigen::Ref<Robbie::Matrix<scalar>> var, Eigen::Ref<Robbie::Matrix<scalar>> grad ) override
    {
        Optimizer<scalar>::register_variable( var, grad );
        m_matrix.push_back( Matrix<scalar>::Zero( var.rows(), var.cols() ) );
        v_matrix.push_back( Matrix<scalar>::Zero( var.rows(), var.cols() ) );
    }

    void clear() override
    {
        Optimizer<scalar>::clear();
        m_matrix.clear();
        v_matrix.clear();
        timestep = 0;
    }

    Adam() = default;
    Adam( scalar alpha ) : alpha( alpha ) {}
    Adam( scalar alpha, scalar beta1, scalar beta2, scalar epsilon )
            : alpha( alpha ), beta1( beta1 ), beta2( beta2 ), epsilon( epsilon )
    {
    }

    void optimize() override
    {
        scalar beta_1_t = std::pow( beta1, timestep + 1 );
        scalar beta_2_t = std::pow( beta2, timestep + 1 );
        scalar alpha_t  = alpha * std::sqrt( 1.0 - beta_2_t ) / ( 1.0 - beta_1_t );

        for( size_t iv = 0; iv < this->variables.size(); iv++ )
        {
            // Update first moments
            m_matrix[iv] = m_matrix[iv] * beta1 + ( 1.0 - beta1 ) * ( this->gradients[iv] );
            // Update second moments
            v_matrix[iv] = v_matrix[iv] * beta2 + ( 1.0 - beta2 ) * this->gradients[iv].array().pow( 2 ).matrix();
            this->variables[iv]
                -= alpha_t * ( m_matrix[iv].array() / ( v_matrix[iv].array().sqrt() + epsilon ) ).matrix();
        }
    }
};

} // namespace Robbie::Optimizers