#pragma once

#include "util/common.hpp"

namespace ESN
{

using namespace UTIL;

// Generates sparse matrix
template <Weight T, RandomNumberEngine Generator = std::mersenne_twister_engine<
                        unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff,
                        7, 0x9d2c5680, 15, 0xefc60000, 18, 1812433253>>
constexpr inline SMat<T>
generate_sparse(
    const Index rows, const Index cols, const T sparsity,
    const typename Generator::result_type seed =
        typename Generator::result_type{ 0 },
    const std::function<T( const T, Generator & )> & gen_value =
        []( [[maybe_unused]] const T x, [[maybe_unused]] Generator & y ) {
            return T{ 1. };
        } ) {
    if ( T{ 0. } > sparsity || T{ 1. } < sparsity ) {
        std::cerr << std::format(
            "Matrix sparsity must satisfy: 0 <= sparsity <= 1 "
            "(sparsity = "
            "{})\n",
            sparsity );
        exit( EXIT_FAILURE );
    }

    // Get random value generator & distribution generator
    auto gen{ Generator( seed ) };
    gen.discard( 100 );
    auto distribution{ std::uniform_real_distribution<T>( 0.0, 1.0 ) };

    // Storage for triplets reserved based on estimated no. of elements
    std::vector<Eigen::Triplet<T, Index>> triplets(
        static_cast<std::size_t>( static_cast<T>( rows * cols ) * sparsity ) );
    for ( Index i{ 0 }; i < rows; ++i ) {
        for ( Index j{ 0 }; j < cols; ++j ) {
            const auto x{ distribution( gen ) };
            if ( x < sparsity ) {
                triplets.push_back(
                    Eigen::Triplet{ i, j, gen_value( x, gen ) } );
            }
        }
    }

    SMat<T> result( rows, cols );
    result.setFromTriplets( triplets.cbegin(), triplets.cend() );
    return result;
}

} // namespace ESN
