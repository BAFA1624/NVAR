#pragma once

#include "util/common.hpp"

#include <boost/math/distributions.hpp>
#include <boost/random.hpp>
#include <numeric>

namespace ESN
{

using namespace UTIL;

// Generates sparse matrix
template <
    Weight T, std::integral Seed_t = std::uint32_t,
    PseudoRandomNumberGenerator<Seed_t> Generator = default_generator<Seed_t>,
    RandomDistribution<Generator> Distribution = boost::random::uniform_01<T>>
constexpr inline SMat<T>
generate_sparse(
    const Index rows, const Index cols, const T sparsity,
    const Seed_t                        seed = Seed_t{ 0 },
    const std::function<T( const T )> & gen_value =
        []( [[maybe_unused]] const T x ) { return T{ 1. }; },
    Distribution distribution = boost::random::uniform_01<T>{} ) {
    // Check valid threshold
    if ( !( 0 <= sparsity && 1 >= sparsity ) ) {
        std::cerr << std::format(
            "Matrix sparsity must be bound by: 0 <= sparsity <= 1 "
            "(sparsity = "
            "{})\n",
            sparsity );
        exit( EXIT_FAILURE );
    }

    // Get random value generator & distribution generator
    auto generator{ Generator( seed ) };

    // Get threshold value based on threshold, & the min, & max of the given
    // distribution
    const T threshold{ boost::math::quantile( distribution, sparsity ) };
    std::cout << std::format(
        "distribution.min() = {}, distribution.max() = {}n\nthreshold = "
        "{}\n",
        distribution.min(), distribution.max(), threshold );

    // Storage for triplets reserved based on estimated no. of elements
    std::vector<Eigen::Triplet<T, Index>> triplets(
        static_cast<std::size_t>( static_cast<T>( rows * cols ) * sparsity ) );
    for ( Index i{ 0 }; i < rows; ++i ) {
        for ( Index j{ 0 }; j < cols; ++j ) {
            if ( const auto x{ distribution( generator ) }; x < threshold ) {
                triplets.push_back( Eigen::Triplet{ i, j, gen_value( x ) } );
            }
        }
    }

    SMat<T> result( rows, cols );
    result.setFromTriplets( triplets.cbegin(), triplets.cend() );
    return result;
}

} // namespace ESN
