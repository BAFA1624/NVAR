#pragma once

#include "util/common.hpp"

// Includes to calculate eigenvalues of sparse matrices
#include "Spectra/GenEigsSolver.h"
#include "Spectra/MatOp/DenseGenMatProd.h"
#include "Spectra/MatOp/SparseGenMatProd.h"

#include <chrono>
#include <random>

namespace ESN
{

// Enum class to describe the initialization scheme for the input weights
enum class input_t : UTIL::Index {
    split = 1,          // Split evenly based on dimensionality of input
    homogeneous = 2,    // Initialise inputs homogeneously
    sparse = 4,         // Use same sparsity as adjacency matrix
    dense = 8,          // Fill all W_in values,
    default_input = 10, // Default: homogeneous | dense
    max
};
ENUM_FLAGS( input_t )

// Enum class to describe the initialisation scheme for the adjacency matrix
enum class adjacency_t : UTIL::Index {
    sparse = 1,
    dense = 2 /* default */,
    max
};
ENUM_FLAGS( adjacency_t )

// Enum class to describe the feature vector construction
enum class feature_t : UTIL::Index {
    reservoir = 1 /* Always required */,
    linear = 2,
    bias = 4,
    default_feature = 7, // bias, linear, & reservoir
    max
};
ENUM_FLAGS( feature_t )


template <UTIL::Weight T, feature_t Feature_init>
class DefaultConstructor
{
    private:
    UTIL::Index m_d;
    UTIL::Index m_n_node;

    constexpr static bool m_feature_bias =
        static_cast<bool>( Feature_init & feature_t::bias );
    constexpr static bool m_feature_linear =
        static_cast<bool>( Feature_init & feature_t::linear );

    [[nodiscard]] constexpr inline UTIL::Index feature_size() const noexcept {
        return ( m_feature_bias ? 1 : 0 ) + ( m_feature_linear ? m_d : 0 )
               + m_n_node;
    }

    public:
    DefaultConstructor( const UTIL::Index d, const UTIL::Index n_nonlin ) :
        m_d( d ), m_n_node( n_nonlin ) {}

    constexpr inline auto
    construct( [[maybe_unused]] const UTIL::ConstRefMat<T> & u,
               const UTIL::ConstRefMat<T> &                  R ) {
        UTIL::Mat<T> features( feature_size(), R.cols() );

        UTIL::Index offset{ 0 };

        // If necessary, add bias to feature matrix
        if constexpr ( m_feature_bias ) {
            features( 0, Eigen::placeholders::all ) =
                UTIL::RowVec<T>::Ones( features.cols() );
            offset++;
        }

        // If necessary, add linear component to feature matrix
        if constexpr ( m_feature_linear ) {
            features( Eigen::seq( offset, offset + m_d - 1 ),
                      Eigen::placeholders::all )
                << UTIL::Mat<T>{ u };
            offset += m_d;
        }

        // Add reservoir states to feature matrix
        features( Eigen::seq( offset, Eigen::placeholders::last ),
                  Eigen::placeholders::all ) = UTIL::Mat<T>{ R };

        return features;
    }
};

// static_assert(
//     UTIL::Constructor<DefaultConstructor<double, feature_t::default_feature>>
//     );

// This function is extremely sensitive to the chosen Generator
template <UTIL::Weight T, bool split = false,
          UTIL::RandomNumberEngine Generator = std::mersenne_twister_engine<
              unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,
              0x9d2c5680, 15, 0xefc60000, 18, 1812433253>>
constexpr inline std::vector<Eigen::Triplet<T, UTIL::Index>>
generate_sparse_triplets(
    const UTIL::Index rows, const UTIL::Index cols, const T sparsity,
    Generator &                                      gen,
    const std::function<T( const T, Generator & )> & gen_value =
        []( [[maybe_unused]] const T x, [[maybe_unused]] Generator & gen ) {
            static auto dist{ std::uniform_real_distribution<T>( -1., 1. ) };
            return dist( gen );
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
    std::vector<Eigen::Triplet<T, UTIL::Index>> triplets(
        static_cast<std::size_t>( static_cast<T>( rows * cols ) * sparsity ) );

    if constexpr ( split ) {
        std::vector<UTIL::Index> sizes( static_cast<std::size_t>( cols ),
                                        rows / cols );
        for ( UTIL::Index i{ 0 }; i < rows % cols; ++i ) {
            sizes[static_cast<std::size_t>( i )]++;
        }

        UTIL::Index offset{ 0 };
        for ( const auto [j, n_rows] : sizes | std::views::enumerate ) {
            for ( UTIL::Index i{ offset }; i < offset + n_rows; ++i ) {
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
        for ( UTIL::Index i{ 0 }; i < rows; ++i ) {
            for ( UTIL::Index j{ 0 }; j < cols; ++j ) {
                const auto x{ distribution( gen ) };
                if ( sparsity == static_cast<T>( 1. ) || x < sparsity ) {
                    triplets.push_back(
                        Eigen::Triplet{ i, j, gen_value( x, gen ) } );
                }
            }
        }
    }

    triplets.shrink_to_fit();
    std::cout << "SPARSENESS: "
              << static_cast<T>( triplets.size() )
                     / static_cast<T>( rows * cols )
              << std::endl;
    return triplets;
}

// This function is extremely sensitive to the chosen Generator
template <UTIL::Weight T, bool split = false,
          UTIL::RandomNumberEngine Generator = std::mersenne_twister_engine<
              unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,
              0x9d2c5680, 15, 0xefc60000, 18, 1812433253>>
constexpr inline std::vector<Eigen::Triplet<T, UTIL::Index>>
generate_sparse_triplets(
    const UTIL::Index rows, const UTIL::Index cols, const T sparsity,
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
template <UTIL::Weight T, bool split = false,
          UTIL::RandomNumberEngine Generator = std::mersenne_twister_engine<
              unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,
              0x9d2c5680, 15, 0xefc60000, 18, 1812433253>,
          Eigen::StorageOptions _Options = Eigen::RowMajor,
          std::signed_integral  _StorageIndex = UTIL::Index>
constexpr inline UTIL::SMat<T, _Options, _StorageIndex>
generate_sparse(
    const UTIL::Index rows, const UTIL::Index cols, const T sparsity,
    Generator &                                      gen,
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

    const auto triplets{ generate_sparse_triplets<T, split, Generator>(
        rows, cols, sparsity, gen, gen_value ) };

    UTIL::SMat<T> result( rows, cols );
    result.setFromTriplets( triplets.cbegin(), triplets.cend() );
    return result;
}

// Generates sparse matrix
template <UTIL::Weight T, bool split = false,
          UTIL::RandomNumberEngine Generator = std::mersenne_twister_engine<
              unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,
              0x9d2c5680, 15, 0xefc60000, 18, 1812433253>,
          Eigen::StorageOptions _Options = Eigen::RowMajor,
          std::signed_integral  _StorageIndex = UTIL::Index>
constexpr inline UTIL::SMat<T, _Options, _StorageIndex>
generate_sparse(
    const UTIL::Index rows, const UTIL::Index cols, const T sparsity,
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

    UTIL::SMat<T> result( rows, cols );
    result.setFromTriplets( triplets.cbegin(), triplets.cend() );
    return result;
}

template <std::floating_point   T,
          Eigen::StorageOptions _Options = Eigen::RowMajor,
          std::signed_integral  _StorageIndex = UTIL::Index>
constexpr inline UTIL::Vec<std::complex<T>>
compute_n_eigenvals(
    const UTIL::ConstRefSMat<T, _Options, _StorageIndex> m,
    const UTIL::Index nev, const UTIL::Index ncv,
    const Spectra::SortRule selection = Spectra::SortRule::LargestMagn,
    const UTIL::Index max_it = 1000, const T tol = 1E-10,
    const Spectra::SortRule sort = Spectra::SortRule::LargestMagn ) {
    // Construct matrix operation object using SparseGenMatProd wrapper class
    Spectra::SparseGenMatProd<T, _Options, _StorageIndex> op( m );

    // Construct eigen solver object,  requesting the largest n eigenvalues
    if ( !( nev + 2 <= ncv && ncv <= m.size() ) ) {
        std::cerr << std::format(
            "When computing sparse matrix eigenvalues with Spectra, the value "
            "of 'ncv' must satisfy: nev + 2 ({}) <= ncv ({}) <= n ({}), where "
            "'n' is the size "
            "of the matrix. A good default is 2 * nev + 1 ({}).\n",
            nev + 2, ncv, m.size(), 2 * nev + 1 );
        exit( EXIT_FAILURE );
    }
    Spectra::GenEigsSolver<
        Spectra::SparseGenMatProd<T, _Options, _StorageIndex>>
        eigs( op, nev, ncv );

    // Initialise & compute
    eigs.init();
    const auto nconv{ eigs.compute( selection, max_it, tol, sort ) };

    const auto                 info{ eigs.info() };
    UTIL::Vec<std::complex<T>> eigenvalues( nconv );

    switch ( info ) {
    case Spectra::CompInfo::Successful: {
        eigenvalues = eigs.eigenvalues();
    } break;
    case Spectra::CompInfo::NotComputed: {
        std::cerr << std::format(
            "Eigenvalues of sparse matrix not computed yet.\n" );
        exit( EXIT_FAILURE );
    } break;
    case Spectra::CompInfo::NotConverging: {
        std::cerr << std::format(
            "Eigenvalues of sparse matrix did not converge.\n" );
        exit( EXIT_FAILURE );
    } break;
    case Spectra::CompInfo::NumericalIssue: {
        std::cerr << std::format(
            "Eigenvalues of sparse matrix experienced numerical "
            "instability.\n" );
        exit( EXIT_FAILURE );
    } break;
    };

    return eigenvalues;
}

template <std::floating_point T>
constexpr inline UTIL::Vec<std::complex<T>>
compute_n_eigenvals_dense(
    const UTIL::ConstRefMat<T> & m, const UTIL::Index nev,
    const UTIL::Index       ncv,
    const Spectra::SortRule selection = Spectra::SortRule::LargestMagn,
    const UTIL::Index max_it = 1000, const T tol = 1E-10,
    const Spectra::SortRule sort = Spectra::SortRule::LargestMagn ) {
    // Construct matrix operation object using SparseGenMatProd wrapper class
    Spectra::DenseGenMatProd<T, Eigen::ColMajor> op( m );

    // Construct eigen solver object,  requesting the largest n eigenvalues
    if ( !( nev + 2 <= ncv && ncv <= m.size() ) ) {
        std::cerr << std::format(
            "When computing sparse matrix eigenvalues with Spectra, the value "
            "of 'ncv' must satisfy: nev + 2 ({}) <= ncv ({}) <= n ({}), where "
            "'n' is the size "
            "of the matrix. A good default is 2 * nev + 1 ({}).\n",
            nev + 2, ncv, m.size(), 2 * nev + 1 );
        exit( EXIT_FAILURE );
    }
    Spectra::GenEigsSolver<Spectra::DenseGenMatProd<T, Eigen::ColMajor>> eigs(
        op, static_cast<int>( nev ), static_cast<int>( ncv ) );

    // Initialise & compute
    eigs.init();
    const auto nconv{ eigs.compute( selection, static_cast<int>( max_it ), tol,
                                    sort ) };

    const auto                 info{ eigs.info() };
    UTIL::Vec<std::complex<T>> eigenvalues( nconv );

    switch ( info ) {
    case Spectra::CompInfo::Successful: {
        eigenvalues = eigs.eigenvalues();
    } break;
    case Spectra::CompInfo::NotComputed: {
        std::cerr << std::format(
            "Eigenvalues of dense matrix not computed yet.\n" );
        exit( EXIT_FAILURE );
    } break;
    case Spectra::CompInfo::NotConverging: {
        std::cerr << std::format(
            "Eigenvalues of dense matrix did not converge.\n" );
        exit( EXIT_FAILURE );
    } break;
    case Spectra::CompInfo::NumericalIssue: {
        std::cerr << std::format(
            "Eigenvalues of dense matrix experienced numerical "
            "instability.\n" );
        exit( EXIT_FAILURE );
    } break;
    };

    return eigenvalues;
}

} // namespace ESN
