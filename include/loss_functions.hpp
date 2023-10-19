#pragma once
#include "defines.hpp"

namespace Robbie::LossFunctions
{

template<typename scalar>
class MeanSquareError
{
public:
    static scalar f( const Vector<scalar> & y_true, const Vector<scalar> & y_pred )
    {
        return ( y_true - y_pred ).array().pow( 2 ).mean();
    }

    static Vector<scalar> df( const Vector<scalar> & y_true, const Vector<scalar> & y_pred )
    {
        return 2.0 * ( y_pred - y_true ) / y_true.size();
    }
};

} // namespace Robbie::LossFunctions