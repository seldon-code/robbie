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
        auto greater_than_zero = []( typename Derived::Scalar x ) { return std::max( x, 0.0 ); };
        return x.array().unaryExpr( greater_than_zero );
    }

    template<typename Derived>
    static auto df( const Eigen::MatrixBase<Derived> & x )
    {
        auto one_if_greater_than_zero = []( typename Derived::Scalar x ) { return x >= 0.0 ? 1.0 : 0.0; };
        return x.array().unaryExpr( one_if_greater_than_zero );
    }
};

} // namespace Robbie::ActivationFunctions