#include "mnist_loader.h"
#include <assert.h>
#include <filesystem>
#include <fstream>

mnist_loader::mnist_loader( std::string image_file, std::string label_file, int num )
        : m_size( 0 ), m_rows( 0 ), m_cols( 0 )
{
    load_images( image_file, num );
    load_labels( label_file, num );
}

mnist_loader::mnist_loader( std::string image_file, std::string label_file ) : mnist_loader( image_file, label_file, 0 )
{
    // empty
}

mnist_loader::~mnist_loader()
{
    // empty
}

int mnist_loader::to_int( char * p )
{
    return ( ( p[0] & 0xff ) << 24 ) | ( ( p[1] & 0xff ) << 16 ) | ( ( p[2] & 0xff ) << 8 ) | ( ( p[3] & 0xff ) << 0 );
}

void mnist_loader::load_images( std::string image_file, int num )
{
    std::filesystem::path path( image_file );
    assert( std::filesystem::exists( path ) );

    std::ifstream ifs( image_file.c_str(), std::ios::in | std::ios::binary );
    char p[4];

    ifs.read( p, 4 );
    int magic_number = to_int( p );
    assert( magic_number == 0x803 );

    ifs.read( p, 4 );
    m_size = to_int( p );
    // limit
    if( num != 0 && num < m_size )
        m_size = num;

    ifs.read( p, 4 );
    m_rows = to_int( p );

    ifs.read( p, 4 );
    m_cols = to_int( p );

    char * q = new char[m_rows * m_cols];
    for( int i = 0; i < m_size; ++i )
    {
        ifs.read( q, m_rows * m_cols );
        std::vector<double> image( m_rows * m_cols );
        for( int j = 0; j < m_rows * m_cols; ++j )
        {
            image[j] = q[j] / 255.0;
        }
        m_images.push_back( image );
    }
    delete[] q;

    ifs.close();
}

void mnist_loader::load_labels( std::string label_file, int num )
{
    std::filesystem::path path( label_file );
    assert( std::filesystem::exists( path ) );

    std::ifstream ifs( label_file.c_str(), std::ios::in | std::ios::binary );
    char p[4];

    ifs.read( p, 4 );
    int magic_number = to_int( p );
    assert( magic_number == 0x801 );

    ifs.read( p, 4 );
    int size = to_int( p );
    // limit
    if( num != 0 && num < m_size )
        size = num;

    for( int i = 0; i < size; ++i )
    {
        ifs.read( p, 1 );
        int label = p[0];
        m_labels.push_back( label );
    }

    ifs.close();
}
