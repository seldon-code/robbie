#pragma once
#include "defines.hpp"
#include "optimizer.hpp"
#include <vector>

namespace Robbie
{

template<typename scalar>
class StochasticGradientDescent : public Optimizer<scalar>
{
private:
    scalar learning_rate = 0.0;

public:
    StochasticGradientDescent( scalar learning_rate ) : learning_rate( learning_rate ) {}

    void optimize(
        std::vector<Eigen::Ref<Matrix<scalar>>> & variables,
        std::vector<Eigen::Ref<Matrix<scalar>>> & gradients ) override
    {
        for( size_t i = 0; i < variables.size(); i++ )
        {
            variables[i] -= learning_rate * gradients[i];
        }
    }
};
} // namespace Robbie