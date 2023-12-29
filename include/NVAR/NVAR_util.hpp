#pragma once

#include "util/common.hpp"

#include <cassert>
#include <format>
#include <iostream>

namespace NVAR
{

// Nonlinearity types
enum class nonlinear_t { poly, exp, polyexp };

// Calculates factorial for 0 <= n <= 20

constexpr inline UTIL::Index
factorial_20( const UTIL::Index N ) {
    assert( N >= 0 && N < 21 );
    if ( N >= 21 ) {
        std::cerr << std::format( "Attempted factorial_20({}).", N )
                  << std::endl;
        exit( EXIT_FAILURE );
    }
    if ( N == 0 ) {
        return UTIL::Index{ 1 };
    }
    UTIL::Index val{ 1 };
    for ( UTIL::Index i{ 1 }; i <= N; ++i ) { val *= i; }
    return val;
}

template <nonlinear_t Nonlinearity>
constexpr inline UTIL::Index
def_nonlinear_size( const UTIL::Index d, const UTIL::Index k,
                    const UTIL::Index p ) {
    if constexpr ( Nonlinearity == nonlinear_t::exp ) {
        return d * k;
    }
    else {
        if ( p == 0 ) {
            return UTIL::Index{ 0 };
        }
        assert( d > 0 && k > 0 && p > 0 );
        return factorial_20( d * k + p - 1 )
               / ( factorial_20( p ) * factorial_20( d * k - 1 ) );
    }
}

template <nonlinear_t Nonlinearity>
constexpr inline UTIL::Index
def_total_size( const UTIL::Index d, const UTIL::Index k, const UTIL::Index p,
                const bool constant = true ) {
    return def_nonlinear_size<Nonlinearity>( d, k, p ) + d * k
           + ( constant ? 1 : 0 );
}

constexpr inline std::vector<std::vector<UTIL::Index>>
combinations_with_replacement_indices( const UTIL::Index n,
                                       const UTIL::Index p ) {
    if ( p == 0 ) {
        return {};
    }

    std::size_t              count{ 0 };
    std::vector<UTIL::Index> indices( static_cast<std::size_t>( p ), 0 );

    std::vector<std::vector<UTIL::Index>> result( static_cast<std::size_t>(
        def_nonlinear_size<nonlinear_t::poly>( n, 1, p ) ) );

    while ( true ) {
        // Add current set of indices
        result[count++] = indices;

        // Find rightmost index to increment
        auto j = p - 1;
        while ( j >= 0 && indices[static_cast<std::size_t>( j )] == n - 1 ) {
            j--;
        }

        // If no index found, break out
        if ( j < 0 ) {
            break;
        }

        // Increment found index & adjust subsequent
        indices[static_cast<std::size_t>( j )]++;
        for ( UTIL::Index i{ j + 1 }; i < p; ++i ) {
            indices[static_cast<std::size_t>( i )] =
                indices[static_cast<std::size_t>( i - 1 )];
        }
    }

    return result;
}

template <UTIL::Weight T>
constexpr inline UTIL::Mat<T>
apply_indices( const UTIL::ConstRefMat<T> &                  m,
               const std::vector<std::vector<UTIL::Index>> & indices ) {
    auto result{ UTIL::Mat<T>::Ones( static_cast<UTIL::Index>( indices.size() ),
                                     m.cols() ) };
    const auto calculate_values{ [&indices, &m]( const UTIL::Index i,
                                                 const UTIL::Index j ) -> T {
        T      val{ 1. };
        for ( const auto & idx : indices[i] ) { val *= m( idx, j ); }
        return val;
    } };
    return UTIL::Mat<T>::NullaryExpr(
        static_cast<UTIL::Index>( indices.size() ), m.cols(),
        calculate_values );
}

template <UTIL::Weight T>
constexpr inline UTIL::Vec<T>
combinations_with_replacement( const UTIL::ConstRefVec<T> & v,
                               const UTIL::Index n, const UTIL::Index p ) {
    if ( p == 0 ) {
        return UTIL::Vec<T>{};
    }

    UTIL::Index count{ 0 };

    std::vector<UTIL::Index> indices( static_cast<std::size_t>( p ), 0 );
    UTIL::Vec<T>             result{ UTIL::Vec<T>::Ones(
        def_nonlinear_size<nonlinear_t::poly>( n, 1, p ) ) };

    while ( true ) {
        // Add current result
        for ( const auto i : indices ) { result( count ) *= v( i ); }
        count++;

        // Find rightmost index to increment
        auto j = p - 1;
        while ( j >= 0 && indices[static_cast<std::size_t>( j )] == n - 1 ) {
            j--;
        }

        // If no index found, break out
        if ( j < 0 ) {
            break;
        }

        // Increment found index & adjust subsequent
        indices[static_cast<std::size_t>( j )]++;
        for ( UTIL::Index i{ j + 1 }; i < p; ++i ) {
            indices[static_cast<std::size_t>( i )] =
                indices[static_cast<std::size_t>( i - 1 )];
        }
    }

    return result;
}

template <UTIL::Weight T>
constexpr inline UTIL::Vec<T>
construct_x_i( const UTIL::ConstRefMat<T> & inputs, const UTIL::Index i,
               const UTIL::Index k, const UTIL::Index s ) {
    assert( i + ( k - 1 ) * s < inputs.rows() );
    return inputs( Eigen::seqN( i + ( k - 1 ) * s, k, -s ),
                   Eigen::placeholders::all )
        .template reshaped<Eigen::RowMajor>();
}
template <UTIL::Weight T>
constexpr inline UTIL::Vec<T>
construct_x_i( const UTIL::ConstRefMat<T> & input, const UTIL::Index k,
               const UTIL::Index s ) {
    assert( input.rows() == s * ( k - 1 ) + 1 );
    return input( Eigen::seqN( s * ( k - 1 ), k, -s ),
                  Eigen::placeholders::all )
        .template reshaped<Eigen::RowMajor>();
}

template <UTIL::Weight T>
constexpr inline UTIL::Mat<T>
cycle_inputs( const UTIL::ConstRefMat<T> & prev_input,
              const UTIL::ConstRefMat<T> & new_value ) {
    const UTIL::Index n{ prev_input.rows() };
    auto result = UTIL::Mat<T>( prev_input.rows(), prev_input.cols() );
    result.bottomRows( n - 1 ) = prev_input.topRows( n - 1 );
    result.topRows( 1 ) = new_value;
    return result;
}


constexpr inline UTIL::Index
warmup_offset( const UTIL::Index k, const UTIL::Index s ) {
    return s * ( k - 1 );
}

} // namespace NVAR
