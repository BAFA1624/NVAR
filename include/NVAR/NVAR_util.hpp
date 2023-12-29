#pragma once

#include "util/common.hpp"

#include <cassert>
#include <format>
#include <iostream>

namespace NVAR
{

using namespace UTIL;

// Nonlinearity types
enum class nonlinear_t { poly, exp, polyexp };

// Calculates factorial for 0 <= n <= 20

constexpr inline Index
factorial_20( const Index N ) {
    assert( N >= 0 && N < 21 );
    if ( N >= 21 ) {
        std::cerr << std::format( "Attempted factorial_20({}).", N )
                  << std::endl;
        exit( EXIT_FAILURE );
    }
    if ( N == 0 ) {
        return Index{ 1 };
    }
    Index val{ 1 };
    for ( Index i{ 1 }; i <= N; ++i ) { val *= i; }
    return val;
}

template <nonlinear_t Nonlinearity>
constexpr inline Index
def_nonlinear_size( const Index d, const Index k, const Index p ) {
    if constexpr ( Nonlinearity == nonlinear_t::exp ) {
        return d * k;
    }
    else {
        if ( p == 0 ) {
            return Index{ 0 };
        }
        assert( d > 0 && k > 0 && p > 0 );
        return factorial_20( d * k + p - 1 )
               / ( factorial_20( p ) * factorial_20( d * k - 1 ) );
    }
}

template <nonlinear_t Nonlinearity>
constexpr inline Index
def_total_size( const Index d, const Index k, const Index p,
                const bool constant = true ) {
    return def_nonlinear_size<Nonlinearity>( d, k, p ) + d * k
           + ( constant ? 1 : 0 );
}

constexpr inline std::vector<std::vector<Index>>
combinations_with_replacement_indices( const Index n, const Index p ) {
    if ( p == 0 ) {
        return {};
    }

    std::size_t        count{ 0 };
    std::vector<Index> indices( static_cast<std::size_t>( p ), 0 );

    std::vector<std::vector<Index>> result( static_cast<std::size_t>(
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
        for ( Index i{ j + 1 }; i < p; ++i ) {
            indices[static_cast<std::size_t>( i )] =
                indices[static_cast<std::size_t>( i - 1 )];
        }
    }

    return result;
}

template <Weight T>
constexpr inline Mat<T>
apply_indices( const ConstRefMat<T> &                  m,
               const std::vector<std::vector<Index>> & indices ) {
    auto       result{ Mat<T>::Ones( static_cast<Index>( indices.size() ),
                                     m.cols() ) };
    const auto calculate_values{ [&indices, &m]( const Index i,
                                                 const Index j ) -> T {
        T      val{ 1. };
        for ( const auto & idx : indices[i] ) { val *= m( idx, j ); }
        return val;
    } };
    return Mat<T>::NullaryExpr( static_cast<Index>( indices.size() ), m.cols(),
                                calculate_values );
}

template <Weight T>
constexpr inline Vec<T>
combinations_with_replacement( const ConstRefVec<T> & v, const Index n,
                               const Index p ) {
    if ( p == 0 ) {
        return Vec<T>{};
    }

    Index count{ 0 };

    std::vector<Index> indices( static_cast<std::size_t>( p ), 0 );
    Vec<T>             result{ Vec<T>::Ones(
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
        for ( Index i{ j + 1 }; i < p; ++i ) {
            indices[static_cast<std::size_t>( i )] =
                indices[static_cast<std::size_t>( i - 1 )];
        }
    }

    return result;
}

template <Weight T>
constexpr inline Vec<T>
construct_x_i( const ConstRefMat<T> & inputs, const Index i, const Index k,
               const Index s ) {
    assert( i + ( k - 1 ) * s < inputs.rows() );
    return inputs( Eigen::seqN( i + ( k - 1 ) * s, k, -s ),
                   Eigen::placeholders::all )
        .template reshaped<Eigen::RowMajor>();
}
template <Weight T>
constexpr inline Vec<T>
construct_x_i( const ConstRefMat<T> & input, const Index k, const Index s ) {
    assert( input.rows() == s * ( k - 1 ) + 1 );
    return input( Eigen::seqN( s * ( k - 1 ), k, -s ),
                  Eigen::placeholders::all )
        .template reshaped<Eigen::RowMajor>();
}

template <Weight T>
constexpr inline Mat<T>
cycle_inputs( const ConstRefMat<T> & prev_input,
              const ConstRefMat<T> & new_value ) {
    const Index n{ prev_input.rows() };
    auto        result = Mat<T>( prev_input.rows(), prev_input.cols() );
    result.bottomRows( n - 1 ) = prev_input.topRows( n - 1 );
    result.topRows( 1 ) = new_value;
    return result;
}


constexpr inline Index
warmup_offset( const Index k, const Index s ) {
    return s * ( k - 1 );
}

} // namespace NVAR
