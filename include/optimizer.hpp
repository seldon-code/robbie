#pragma once
#include "defines.hpp"
#include <vector>

namespace Robbie
{

template<typename scalar>
class Optimizer
{
public:
    Optimizer()          = default;
    virtual ~Optimizer() = default;

    virtual void optimize(
        std::vector<Eigen::Ref<Robbie::Matrix<scalar>>> & variables,
        const std::vector<Eigen::Ref<Matrix<scalar>>> & gradients )
        = 0;
};
} // namespace Robbie