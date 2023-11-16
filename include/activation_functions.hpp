#pragma once
#include "defines.hpp"

namespace Robbie::ActivationFunctions
{

class Tanh
{
public:
    template<typename Derived>
    static auto f( const Eigen::MatrixBase<Derived> & x )
    {
        return x.array().tanh();
    }

    template<typename Derived>
    static auto df( const Eigen::MatrixBase<Derived> & x )
    {
        return -x.array().tanh().pow( 2 ) + 1.0;
    }
};

class ReLU
{
public:
    template<typename Derived>
    static auto f( const Eigen::MatrixBase<Derived> & x )
    {
        // The static_cast is necessary to make different scalar types than double compile
        using scalar           = typename Derived::Scalar;
        auto greater_than_zero = []( scalar x ) { return std::max<scalar>( x, static_cast<scalar>( 0.0 ) ); };
        return x.array().unaryExpr( greater_than_zero );
    }

    template<typename Derived>
    static auto df( const Eigen::MatrixBase<Derived> & x )
    {
        // The static_cast is necessary to make different scalar types than double compile
        using scalar                  = typename Derived::Scalar;
        auto one_if_greater_than_zero = []( scalar x ) { return static_cast<scalar>( x >= 0.0 ? 1.0 : 0.0 ); };
        return x.array().unaryExpr( one_if_greater_than_zero );
    }
};

class Softmax
{
public:
    template<typename Derived>
    static auto f( const Eigen::MatrixBase<Derived> & x )
    {
        // Shift by max(x) to avoid numerical problems with large exponents
        auto sum = ( x.array() - x.maxCoeff() ).exp().sum();
        return ( x.array() - x.maxCoeff() ).exp() / sum;
    }

    template<typename Derived>
    static auto df( const Eigen::MatrixBase<Derived> & x )
    {
        auto sum = ( x.array() - x.maxCoeff() ).exp().sum();
        return 1.0 / sum * ( 1.0 - 1.0 / sum ) * ( x.array() - x.maxCoeff() ).exp();
    }
};

} // namespace Robbie::ActivationFunctions