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

} // namespace Robbie::LossFunctions