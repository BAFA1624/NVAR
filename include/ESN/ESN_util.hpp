#pragma once

#include "util/common.hpp"

#include <chrono> // TODO: Remove
#include <random>

namespace ESN
{

using namespace UTIL;

// Enum class to describe the initialization scheme for the input weights
enum class input_init_t : Index {
    split = 1,        // Split evenly based on dimensionality of input
    homogeneous = 2,  // Initialise inputs homogeneously
    sparse = 4,       // Use same sparsity as adjacency matrix
    dense = 8,        // Fill all W_in values
    init_default = 6, // Default: homogeneous | sparse
};

ENUM_FLAGS( input_init_t );

// This function is extremely sensitive to the chosen Generator
template <Weight T, bool split = false,
          RandomNumberEngine Generator = std::mersenne_twister_engine<
              unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,
              0x9d2c5680, 15, 0xefc60000, 18, 1812433253>>
constexpr inline std::vector<Eigen::Triplet<T, Index>>
generate_sparse_triplets(
    const Index rows, const Index cols, const T sparsity, Generator & gen,
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

    //  distribution generator
    auto distribution{ std::uniform_real_distribution<T>( 0.0, 1.0 ) };

    // Storage for triplets reserved based on estimated no. of elements
    std::vector<Eigen::Triplet<T, Index>> triplets(
        static_cast<std::size_t>( static_cast<T>( rows * cols ) * sparsity ) );

    if constexpr ( split ) {
        std::vector<Index> sizes( static_cast<std::size_t>( cols ),
                                  rows / cols );
        for ( Index i{ 0 }; i < rows % cols; ++i ) {
            sizes[static_cast<std::size_t>( i )]++;
        }

        std::cout << std::format( "Splitting {} rows into {} chunks:\n", rows,
                                  cols );
        for ( const auto sz : sizes ) {
            std::cout << "\t- " << sz << std::endl;
        }

        Index offset{ 0 };
        for ( const auto [j, n_rows] : sizes | std::views::enumerate ) {
            for ( Index i{ offset }; i < offset + n_rows; ++i ) {
                const auto x{ distribution( gen ) };
                if ( sparsity == static_cast<T>( 1. ) || x < sparsity ) {
                    triplets.push_back(
                        Eigen::Triplet{ i, j, gen_value( x, gen ) } );
                }
            }
            offset += n_rows;
        }
    }
    else {
        for ( Index i{ 0 }; i < rows; ++i ) {
            for ( Index j{ 0 }; j < cols; ++j ) {
                const auto x{ distribution( gen ) };
                if ( sparsity == static_cast<T>( 1. ) || x < sparsity ) {
                    triplets.push_back(
                        Eigen::Triplet{ i, j, gen_value( x, gen ) } );
                }
            }
        }
    }

    triplets.shrink_to_fit();
    return triplets;
}

// This function is extremely sensitive to the chosen Generator
template <Weight T, bool split = false,
          RandomNumberEngine Generator = std::mersenne_twister_engine<
              unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,
              0x9d2c5680, 15, 0xefc60000, 18, 1812433253>>
constexpr inline std::vector<Eigen::Triplet<T, Index>>
generate_sparse_triplets(
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

    // Get random value generator, & distribution generator
    auto gen{ Generator( seed ) };
    gen.discard( static_cast<unsigned long long>( gen() ) );
    auto distribution{ std::uniform_real_distribution<T>( 0.0, 1.0 ) };

    return generate_sparse_triplets<T, split, Generator>( rows, cols, sparsity,
                                                          gen, gen_value );
}

// Generates sparse matrix
template <Weight T, bool split = false,
          RandomNumberEngine Generator = std::mersenne_twister_engine<
              unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,
              0x9d2c5680, 15, 0xefc60000, 18, 1812433253>>
constexpr inline SMat<T>
generate_sparse(
    const Index rows, const Index cols, const T sparsity, Generator & gen,
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

    const auto triplets_start{ std::chrono::steady_clock::now() };
    const auto triplets{ generate_sparse_triplets<T, split, Generator>(
        rows, cols, sparsity, gen, gen_value ) };
    const auto triplets_finish{ std::chrono::steady_clock::now() };
    std::cout << std::format(
        "Generating triplets took: {}\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(
            triplets_finish - triplets_start ) );

    SMat<T> result( rows, cols );
    result.setFromTriplets( triplets.cbegin(), triplets.cend() );
    return result;
}

// Generates sparse matrix
template <Weight T, bool split = false,
          RandomNumberEngine Generator = std::mersenne_twister_engine<
              unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,
              0x9d2c5680, 15, 0xefc60000, 18, 1812433253>>
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

    const auto triplets_start{ std::chrono::steady_clock::now() };
    const auto triplets{ generate_sparse_triplets<T, split, Generator>(
        rows, cols, sparsity, seed, gen_value ) };
    const auto triplets_finish{ std::chrono::steady_clock::now() };
    std::cout << std::format(
        "Generating triplets took: {}\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(
            triplets_finish - triplets_start ) );

    SMat<T> result( rows, cols );
    result.setFromTriplets( triplets.cbegin(), triplets.cend() );
    return result;
}

} // namespace ESN
