#pragma once

#include "defines.hpp"
#include "layer.hpp"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <random>

namespace Robbie
{
template<typename scalar>
class FCLayer : public Layer<scalar>
{
protected:
    Matrix<scalar> weights;
    Matrix<scalar> weights_error;

    Matrix<scalar> bias;
    Matrix<scalar> bias_error;

public:
    FCLayer( size_t input_size, size_t output_size )
            : Layer<scalar>( input_size, output_size ),
              weights( Matrix<scalar>( output_size, input_size ) ),
              weights_error( Matrix<scalar>( output_size, input_size ) ),
              bias( Vector<scalar>( output_size ) ),
              bias_error( Vector<scalar>( output_size ) )
    {
        auto rd   = std::random_device();
        auto gen  = std::mt19937( rd() );
        auto dist = std::uniform_real_distribution<scalar>( -1, 1 );

        const auto random_lambda = [&]( [[maybe_unused]] scalar x ) { return dist( gen ); };

        weights = weights.array().unaryExpr( random_lambda ) / std::sqrt( input_size );
        bias    = bias.array().unaryExpr( random_lambda ) / std::sqrt( input_size );
    }

    std::string name() override
    {
        return "Fully Connected";
    }

    // returns output for a given input
    Matrix<scalar> forward_propagation( const Matrix<scalar> & input_data ) override
    {
        this->input = input_data;
        this->output
            = ( weights * input_data ).colwise()
              + bias.col(
                  0 ); // We use .col(0), so the bias can be treated as a matrix with fixed columns at compile time
        return this->output;
    }

    // computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    Matrix<scalar> backward_propagation( const Matrix<scalar> & output_error ) override
    {
        auto input_error = weights.transpose() * output_error;
        weights_error    = output_error * this->input.transpose() / output_error.cols();
        bias_error       = ( output_error ).rowwise().mean();
        return input_error;
    }

    // Return the number of trainable parameters
    size_t get_trainable_params() override
    {
        return this->weights.size() + this->bias.size();
    }

    // Get ref to trainable parameters
    std::vector<Eigen::Ref<Matrix<scalar>>> variables() override
    {
        return std::vector<Eigen::Ref<Matrix<scalar>>>{ Eigen::Ref<Matrix<scalar>>( weights ),
                                                        Eigen::Ref<Matrix<scalar>>( bias ) };
    };

    // Get ref to trainable parameters
    std::vector<Eigen::Ref<Matrix<scalar>>> gradients() override
    {
        return std::vector<Eigen::Ref<Matrix<scalar>>>{ Eigen::Ref<Matrix<scalar>>( weights_error ),
                                                        Eigen::Ref<Matrix<scalar>>( bias_error ) };
    };

    // Access the current weights
    Matrix<scalar> get_weights()
    {
        return this->weights;
    }
};
} // namespace Robbie