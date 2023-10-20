#pragma once
#include "defines.hpp"

namespace Robbie
{

/*
Compute the derivatives of a univaritate scalar function at multiple points with central finite differences
*/
template<typename scalar, typename Func>
inline Vector<scalar> finite_diff( Func f, const Vector<scalar> & X, scalar epsilon = 1e-8 )
{
    Vector<scalar> result( X.size() );
    Vector<scalar> diff = Robbie::Vector<scalar>::Zero( X.size() );

    for( size_t dim = 0; dim < X.size(); dim++ )
    {
        diff.setZero();
        diff[dim]   = epsilon;
        result[dim] = 1.0 / ( 2.0 * epsilon ) * ( f( X + diff )[dim] - f( X - diff )[dim] );
    }

    return result;
}

/*
Compute the gradient of a multivariate scalar function with central finite differences
*/
template<typename scalar, typename Func>
inline Vector<scalar> finite_difference_gradient( Func f, const Vector<scalar> & X, scalar epsilon = 1e-8 )
{
    Vector<scalar> result( X.size() );
    Vector<scalar> diff = Robbie::Vector<scalar>::Zero( X.size() );

    for( size_t dim = 0; dim < X.size(); dim++ )
    {
        diff.setZero();
        diff[dim]   = epsilon;
        result[dim] = 1.0 / ( 2.0 * epsilon ) * ( f( X + diff ) - f( X - diff ) );
    }

    return result;
}

} // namespace Robbie
