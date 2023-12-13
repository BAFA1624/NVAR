#pragma once

#include "util/common.hpp"

#include <numeric>

namespace ESN
{

using namespace UTIL;

// Generates sparse matrix
template <Weight T, std::unsigned_integral Seed_t = std::uint_fast32_t,
          typename result_type = T,
          RandomNumberDistribution Dist = std::uniform_real_distribution<T>,
          std::uniform_random_bit_generator Generator =
              std::mersenne_twister_engine<Seed_t, 32, 624, 397, 31, 0x9908b0df,
                                           11, 0xffffffff, 7, 0x9d2c5680, 15,
                                           0xefc60000, 18, 1812433253>>
constexpr inline SMat<T>
generate_sparse(
    const Index rows, const Index cols, const T sparsity,
    const Seed_t                        seed = Seed_t{ 0 },
    const std::function<T( const T )> & gen_value =
        []( [[maybe_unused]] const T x ) { return T{ 1. }; },
    const std::function<T( const T, const T, const T )> & threshold_func =
        []( const T min, const T max, const T sparsity ) {
            return min + sparsity * ( max - min );
        },
    const T param_a = T{ 0. }, const T param_b = T{ 1. } ) {
    // Check valid threshold
    if ( !( 0 <= sparsity && 1 >= sparsity ) ) {
        std::cerr << std::format(
            "Matrix sparsity must be bound by: 0 <= sparsity <= 1 (sparsity = "
            "{})\n",
            sparsity );
        exit( EXIT_FAILURE );
    }

    // Get random value generator & distribution generator
    auto gen{ Generator( seed ) };
    auto distribution = Dist( param_a, param_b );

    // Get threshold value based on threshold, & the min, & max of the given
    // distribution
    const T threshold{ threshold_func( param_a, param_b, sparsity ) };
    std::cout << std::format(
        "distribution.min() = {}, distribution.max() = {}n\nthreshold = {}\n",
        distribution.min(), distribution.max(), threshold );

    // Storage for triplets reserved based on estimated no. of elements
    std::vector<Eigen::Triplet<T, Index>> triplets(
        static_cast<std::size_t>( static_cast<T>( rows * cols ) * sparsity ) );
    for ( Index i{ 0 }; i < rows; ++i ) {
        for ( Index j{ 0 }; j < cols; ++j ) {
            if ( const auto x{ distribution( gen ) }; x < threshold ) {
                triplets.push_back( Eigen::Triplet{ i, j, gen_value( x ) } );
            }
        }
    }

    SMat<T> result( rows, cols );
    result.setFromTriplets( triplets.cbegin(), triplets.cend() );
    return result;
}

} // namespace ESN
