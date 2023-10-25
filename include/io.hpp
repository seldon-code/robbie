#pragma once
#include "defines.hpp"
#include <fstream>

namespace Robbie
{

template<typename Derived>
void matrix_to_file( const std::string & filename, const Eigen::MatrixBase<Derived> & matrix )
{
    std::ofstream file( filename );
    if( file.is_open() )
    {
        file << matrix;
    }
}

} // namespace Robbie