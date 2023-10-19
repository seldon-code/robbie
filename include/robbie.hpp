#pragma once
#include "activation_functions.hpp"
#include "activation_layer.hpp"
#include "defines.hpp"
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
    srand( 1 );

    std::vector<Vector<double>> x_train( 4, Vector<double>( 2 ) );
    x_train[0] << 0, 0;
    x_train[1] << 0, 1;
    x_train[2] << 1, 0;
    x_train[3] << 1, 1;

    std::vector<Vector<double>> y_train( 4, Vector<double>( 1 ) );
    y_train[0] << 0;
    y_train[1] << 1;
    y_train[2] << 1;
    y_train[3] << 0;

    auto network = Network<double, LossFunctions::MeanSquareError<double>>();
    network.add( FCLayer<double>( 2, 3 ) );
    network.add( ActivationLayer<double, ActivationFunctions::Tanh<double>>() );
    network.add( FCLayer<double>( 3, 1 ) );
    network.add( ActivationLayer<double, ActivationFunctions::Tanh<double>>() );

    network.fit( x_train, y_train, 1000, 0.1 );

    auto out = network.predict( x_train );
    fmt::print( "Output = {}\n", out );
}
} // namespace Robbie