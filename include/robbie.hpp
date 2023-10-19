#pragma once
#include "activation_functions.hpp"
#include "activation_layer.hpp"
#include "fc_layer.hpp"
#include "layer.hpp"
#include "loss_functions.hpp"
#include "network.hpp"
#include <fmt/format.h>

namespace Robbie
{
void do_stuff()
{

    auto layer      = FCLayer<double>( 3, 4 );
    auto activation = ActivationLayer<double, ActivationFunctions::Tanh<double>>();

    auto network = Network<double, LossFunctions::MeanSquareError<double>>();
    network.add( FCLayer<double>( 3, 4 ) );

    fmt::print( "doing stuff" );
}
} // namespace Robbie