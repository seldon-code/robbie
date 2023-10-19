#pragma once
#include "defines.hpp"

namespace Robbie::ActivationFunctions
{

template<typename scalar>
class Tanh
{
public:
    static Vector<scalar> f( Vector<scalar> x )
    {
        return x.array().tanh();
    }

    static Vector<scalar> df( Vector<scalar> x )
    {
        return -x.array().tanh().pow( 2 ) + 1.0;
    }
};

} // namespace Robbie::ActivationFunctions