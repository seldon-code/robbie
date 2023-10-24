#pragma once
// #include <eigen3/Eigen/src/Core/util/Constants.h>
#include "defines.hpp"
#include <eigen3/Eigen/Dense>

namespace Robbie
{

template<typename scalar>
class Layer
{
protected:
    Matrix<scalar> input;
    Matrix<scalar> output;

public:
    Layer() = default;

    // computes the output Y of a layer for a given input
    virtual Matrix<scalar> forward_propagation( const Matrix<scalar> & input ) = 0;

    // computes dE/dX for a given dE/dY (and update parameters if any)
    virtual Matrix<scalar> backward_propagation( const Matrix<scalar> & output_error, scalar learning_rate ) = 0;

    // Get trainable parameters
    virtual int get_trainable_params() = 0;

    virtual ~Layer() = default;
};

} // namespace Robbie