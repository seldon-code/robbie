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
    Vector<scalar> input;
    Vector<scalar> output;

public:
    Layer() = default;

    // computes the output Y of a layer for a given input
    virtual Vector<scalar> forward_propagation( const Vector<scalar> & input ) = 0;

    // computes dE/dX for a given dE/dY (and update parameters if any)
    virtual Vector<scalar> backward_propagation( const Vector<scalar> & output_error, scalar learning_rate ) = 0;

    virtual ~Layer() = default;
};

} // namespace Robbie