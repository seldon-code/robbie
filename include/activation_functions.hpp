#pragma once
#include "defines.hpp"

namespace Robbie::ActivationFunctions
{

template<typename scalar>
class Tanh
{
public:
    static Matrix<scalar> f( const Matrix<scalar> & x )
    {
        return x.array().tanh();
    }

    static Matrix<scalar> df( const Matrix<scalar> & x )
    {
        return -x.array().tanh().pow( 2 ) + 1.0;
    }
};

template<typename scalar>
class ReLU
{
public:
    static Matrix<scalar> f( const Matrix<scalar> & x )
    {
        auto greater_than_zero = []( scalar x ) { return std::max( x, 0.0 ); };
        return x.array().unaryExpr( greater_than_zero );
    }

    static Matrix<scalar> df( const Matrix<scalar> & x )
    {
        auto one_if_greater_than_zero = []( scalar x ) { return x >= 0.0 ? 1.0 : 0.0; };
        return x.array().unaryExpr( one_if_greater_than_zero );
    }
};

} // namespace Robbie::ActivationFunctions