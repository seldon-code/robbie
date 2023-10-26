#pragma once
#include "defines.hpp"
#include "optimizer.hpp"
#include "stochastic_gradient_descent.hpp"
#include <eigen3/Eigen/Dense>
#include <memory>
#include <optional>

namespace Robbie
{

template<typename scalar>
class Layer
{
protected:
    Matrix<scalar> input;
    Matrix<scalar> output;
    std::optional<size_t> input_size  = std::nullopt;
    std::optional<size_t> output_size = std::nullopt;
    std::unique_ptr<Optimizer<scalar>> opt;

public:
    Layer() = default;
    Layer( std::optional<size_t> input_size, std::optional<size_t> output_size )
            : input_size( input_size ),
              output_size( output_size ),
              opt( std::make_unique<StochasticGradientDescent<scalar>>( 0.1 ) )
    {
    }

    // TODO: figure out how to implement copy constructor
    // Layer( const Layer & l )
    //         : input( l.input ), output( l.output ), input_size( l.input_size ), output_size( l.output_size )
    // {
    //     opt = std::make_unique<Optimizer<scalar>>( l.opt );
    // }

    virtual std::string name() = 0;

    std::optional<size_t> get_input_size()
    {
        return this->input_size;
    }

    std::optional<size_t> get_output_size()
    {
        return this->output_size;
    }

    // forward propagation of inputs
    virtual Matrix<scalar> forward_propagation( const Matrix<scalar> & input ) = 0;

    // computes the output Y of a layer for a given input (the same as forward propagation except for dropout layers)
    virtual Matrix<scalar> predict( const Matrix<scalar> & input )
    {
        return forward_propagation( input );
    };

    // computes dE/dX for a given dE/dY (and update parameters if any)
    virtual Matrix<scalar> backward_propagation( const Matrix<scalar> & output_error, scalar learning_rate ) = 0;

    // Get trainable parameters
    virtual int get_trainable_params() = 0;

    virtual ~Layer() = default;
};

} // namespace Robbie