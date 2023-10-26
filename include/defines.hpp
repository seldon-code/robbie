#pragma once
#include <eigen3/Eigen/Dense>

namespace Robbie
{
template<typename scalar>
using Vector = Eigen::Vector<scalar, Eigen::Dynamic>;

template<typename scalar>
using Matrix = Eigen::MatrixX<scalar>;

} // namespace Robbie