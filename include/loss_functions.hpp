#pragma once
#include "defines.hpp"

namespace Robbie::LossFunctions
{

// template<typename scalar>
class MeanSquareError
{
public:
    template<typename Derived, typename Derived2>
    static auto f( const Eigen::MatrixBase<Derived> & y_true, const Eigen::MatrixBase<Derived2> & y_pred )
    {
        return ( y_true - y_pred ).array().pow( 2 ).colwise().mean();
    }

    template<typename Derived, typename Derived2>
    static auto df( const Eigen::MatrixBase<Derived> & y_true, const Eigen::MatrixBase<Derived2> & y_pred )
    {
        return 2.0 * ( y_pred - y_true ) / y_true.rows();
    }
};

class SumSquareError
{
public:
    template<typename Derived, typename Derived2>
    static auto f( const Eigen::MatrixBase<Derived> & y_true, const Eigen::MatrixBase<Derived2> & y_pred )
    {
        return ( y_true - y_pred ).array().pow( 2 ).colwise().sum();
    }

    template<typename Derived, typename Derived2>
    static auto df( const Eigen::MatrixBase<Derived> & y_true, const Eigen::MatrixBase<Derived2> & y_pred )
    {
        return 2.0 * ( y_pred - y_true );
    }
};

class CrossEntropy
{
public:
    template<typename Derived, typename Derived2>
    static auto f( const Eigen::MatrixBase<Derived> & y_true, const Eigen::MatrixBase<Derived2> & y_pred )
    {
        constexpr typename Derived::Scalar epsilon = 1e-12;
        return -( y_true.array() * ( y_pred.array() + epsilon ).log2() ).colwise().sum();
    }

    template<typename Derived, typename Derived2>
    static auto df( const Eigen::MatrixBase<Derived> & y_true, const Eigen::MatrixBase<Derived2> & y_pred )
    {
        constexpr typename Derived::Scalar log2e   = 1.44269504089;
        constexpr typename Derived::Scalar epsilon = 1e-12;
        return -( y_true.array() * ( y_pred.array() + epsilon ).pow( -1 ) ).matrix() * log2e;
    }
};

} // namespace Robbie::LossFunctions