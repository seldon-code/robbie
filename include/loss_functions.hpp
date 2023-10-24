#pragma once
#include "defines.hpp"

namespace Robbie::LossFunctions
{

template<typename scalar>
class MeanSquareError
{
public:
    static scalar f( const Matrix<scalar> & y_true, const Matrix<scalar> & y_pred )
    {
        return ( y_true - y_pred ).array().pow( 2 ).mean();
    }

    static Matrix<scalar> df( const Matrix<scalar> & y_true, const Matrix<scalar> & y_pred )
    {
        return 2.0 * ( y_pred - y_true ) / y_true.size();
    }
};

} // namespace Robbie::LossFunctions