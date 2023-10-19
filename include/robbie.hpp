#pragma once
#include "activation_functions.hpp"
#include "activation_layer.hpp"
#include "fc_layer.hpp"
#include "layer.hpp"
#include "loss_functions.hpp"
#include "network.hpp"
#include <fmt/format.h>
#include <fmt/ranges.h>

namespace Robbie
{
void do_stuff()
{
    std::vector<Vector<double>> x_train{
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 },
    };

    std::vector<Vector<double>> y_train{
        Vector<double>{ 0 },
        Vector<double>{ 1 },
        Vector<double>{ 1 },
        Vector<double>{ 0 },
    };

    auto network = Network<double, LossFunctions::MeanSquareError<double>>();
    network.add( FCLayer<double>( 2, 3 ) );
    network.add( ActivationLayer<double, ActivationFunctions::Tanh<double>>() );
    network.add( FCLayer<double>( 3, 1 ) );
    network.add( ActivationLayer<double, ActivationFunctions::Tanh<double>>() );

    network.fit( x_train, y_train, 1000, 0.1 );

    auto out = network.predict( x_train );
    fmt::print( "{}", out );
}
} // namespace Robbie

// import numpy as np

// from network import Network
// from fc_layer import FCLayer
// from activation_layer import ActivationLayer
// from activations import tanh, tanh_prime
// from losses import mse, mse_prime

// # training data
// x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
// y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

// # network
// net = Network()
// net.add(FCLayer(2, 3))
// net.add(ActivationLayer(tanh, tanh_prime))
// net.add(FCLayer(3, 1))
// net.add(ActivationLayer(tanh, tanh_prime))

// # train
// net.use(mse, mse_prime)
// net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

// # test
// out = net.predict(x_train)
// print(out)
// view raw
