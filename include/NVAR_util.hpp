#pragma once

#include "Eigen/Core"
// #include "nlohmann/json.hpp"

// #include <algorithm>
#include <cassert>
#include <complex>
// #include <concepts>
#include <filesystem>
#include <format>
#include <iostream>
#include <map>
#include <ranges>
#include <regex>
#include <tuple>

namespace NVAR
{

template <typename T>
struct is_complex : std::false_type
{};
template <std::floating_point T>
struct is_complex<std::complex<T>> : std::true_type
{};

// Typenames:

// Typedef for all integral types. Is the same as Eigen::Index
using Index = std::ptrdiff_t;

// Concept for all weight types;
template <typename T>
concept Weight = std::floating_point<T> || is_complex<T>::value;

template <Weight T, Index N = Eigen::Dynamic>
using Vec = Eigen::Vector<T, N>;
template <Weight T, Index N = Eigen::Dynamic>
using RowVec = Eigen::RowVector<T, N>;
template <Weight T, Index N = Eigen::Dynamic>
using RefVec = Eigen::Ref<Vec<T, N>>;
template <Weight T, Index N = Eigen::Dynamic>
using RefRowVec = Eigen::Ref<RowVec<T, N>>;
template <Weight T, Index N = Eigen::Dynamic>
using ConstRefVec = Eigen::Ref<const Vec<T, N>>;
template <Weight T, Index N = Eigen::Dynamic>
using ConstRefRowVec = Eigen::Ref<const RowVec<T, N>>;
template <Index N = Eigen::Dynamic>
using Indices = Eigen::Vector<Index, N>;
template <Index N = Eigen::Dynamic>
using RefIndices = Eigen::Ref<Indices<N>>;
template <Weight T, Index R = Eigen::Dynamic, Index C = Eigen::Dynamic>
using Mat = Eigen::Matrix<T, R, C>;
template <Weight T, Index R = Eigen::Dynamic, Index C = Eigen::Dynamic>
using RefMat = Eigen::Ref<Mat<T, R, C>>;
template <Weight T, Index R = Eigen::Dynamic, Index C = Eigen::Dynamic>
using ConstRefMat = Eigen::Ref<const Mat<T, R, C>>;

using FeatureVecShape = std::vector<std::tuple<NVAR::Index, NVAR::Index>>;
template <Weight T, Index R = -1, Index C = -1>
using DataPair = std::tuple<Mat<T, R, C>, Mat<T, R, C>>;

template <typename T, NVAR::Index R = -1, NVAR::Index C = -1>
std::string
mat_shape_str( const NVAR::ConstRefMat<T, R, C> m ) {
    return std::format( "({}, {})", m.rows(), m.cols() );
}

// Nonlinearity types
enum class nonlinear_t { poly, exp, polyexp };

// Calculates factorial for 0 <= n <= 20
template <Index N>
consteval inline Index
factorial_20() {
    static_assert( N >= 0, "Attempted -ve factorial." );
    static_assert( N < 21, "N! for N > 20 causes overflow." );
    if constexpr ( N == Index{ 0 } || N == Index{ 1 } ) {
        return Index{ 1 };
    }
    Index result{ 1 };
    for ( Index i{ 1 }; i <= N; ++i ) { result *= i; }
    return result;
}

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

template <Index d, Index k, Index p, nonlinear_t Nonlinearity>
consteval inline Index
def_nonlinear_size() {
    if ( Nonlinearity == nonlinear_t::exp ) {
        return d * k;
    }
    else {
        if constexpr ( p == 0 ) {
            return Index{ 0 };
        }
        static_assert( d > 0 && k > 0 && p > 0,
                       "NVAR params d, k, & p must be > 0. " );
        /*
         * Nonlinear size given by: (d * k + p - 1)! / (p!(d * k - 1)!)
         */
        return factorial_20<d * k + p - 1>()
               / ( factorial_20<p>() * factorial_20<d * k - 1>() );
    }
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

template <Index d, Index k, Index p, nonlinear_t Nonlinearity,
          bool constant = true>
consteval inline Index
def_total_size() {
    if constexpr ( Nonlinearity == nonlinear_t::exp ) {
        return 2 * d * k + ( constant ? 1 : 0 );
    }
    else {
        static_assert( d > 0 && k > 0 && p >= 0,
                       "NVAR params d, k, must be > 0. p >= 0." );
        return def_nonlinear_size<d, k, p, Nonlinearity>() + d * k
               + ( constant ? 1 : 0 );
    }
}

template <nonlinear_t Nonlinearity>
constexpr inline Index
def_total_size( const Index d, const Index k, const Index p,
                const bool constant = true ) {
    return def_nonlinear_size<Nonlinearity>( d, k, p ) + d * k
           + ( constant ? 1 : 0 );
}

template <Weight T, Index d, Index k, Index p>
constexpr inline Vec<T, def_nonlinear_size<d, k, p, nonlinear_t::poly>()>
combinations_with_replacement( const ConstRefVec<T, d * k> v ) {
    if constexpr ( p == 0 ) {
        return Vec<T, 0>{};
    }

    constexpr Index result_sz{
        def_nonlinear_size<d, k, p, nonlinear_t::poly>()
    };
    constexpr Index n{ d * k };
    Index           count{ 0 };

    std::vector<Index> indices( p, 0 );
    Vec<T, result_sz>  result{ Vec<T, result_sz>::Ones() };

    while ( true ) {
        // Add current result
        for ( const auto i : indices ) { result[count] *= v[i]; }
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
combinations_with_replacement( const ConstRefVec<T> v, const Index d,
                               const Index k, const Index p ) {
    if ( p == 0 ) {
        return Vec<T>{};
    }

    const Index n{ d * k };
    Index       count{ 0 };

    std::vector<Index> indices( static_cast<std::size_t>( p ), 0 );
    Vec<T>             result{ Vec<T>::Ones(
        def_nonlinear_size<nonlinear_t::poly>( d, k, p ) ) };

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

// Implementation from: https://github.com/pconstr/eigen-ridge.git
template <Weight T>
constexpr inline Mat<T>
ridge( const Mat<T> A, const Mat<T> y, const T alpha ) {
    const auto & svd = A.jacobiSvd( Eigen::ComputeFullU | Eigen::ComputeFullV );
    const auto & s = svd.singularValues();
    const auto   r = s.rows();
    const auto & D =
        s.cwiseQuotient( ( s.array().square() + alpha ).matrix() ).asDiagonal();
    const auto factor = svd.matrixV().leftCols( r ) * D
                        * svd.matrixU().transpose().topRows( r );
    return y.transpose() * factor;
}

template <Weight T, Index d, Index k, Index s, Index N>
constexpr inline Mat<T, k, d>
construct_x_i( const ConstRefMat<T, N, d> inputs, const Index i ) {
    assert( i + ( k - 1 ) * s < N );
    return inputs( Eigen::seqN( i + ( k - 1 ) * s, Eigen::fix<k>,
                                Eigen::fix<-s>() ),
                   Eigen::placeholders::all )
        .template reshaped<Eigen::RowMajor>();
}
template <Weight T>
constexpr inline Vec<T>
construct_x_i( const ConstRefMat<T> inputs, const Index i, const Index k,
               const Index s ) {
    assert( i + ( k - 1 ) * s < inputs.rows() );
    return inputs( Eigen::seqN( i + ( k - 1 ) * s, k, -s ),
                   Eigen::placeholders::all )
        .template reshaped<Eigen::RowMajor>();
}
template <Weight T>
constexpr inline Vec<T>
construct_x_i( const ConstRefMat<T> input, const Index k, const Index s ) {
    assert( input.rows() == s * ( k - 1 ) + 1 );
    return input( Eigen::seqN( s * ( k - 1 ), k, -s ),
                  Eigen::placeholders::all )
        .template reshaped<Eigen::RowMajor>();
}
template <Weight T>
constexpr inline Mat<T>
cycle_inputs( const ConstRefMat<T> prev_input,
              const ConstRefMat<T> new_value ) {
    const Index n{ prev_input.rows() };
    auto        result = Mat<T>( prev_input.rows(), prev_input.cols() );
    result.bottomRows( n - 1 ) = prev_input.topRows( n - 1 );
    result.topRows( 1 ) = new_value;
    return result;
}

std::map<std::string, Index> parse_filename( const std::string_view filename );

std::filesystem::path get_filename( const std::vector<Index> & params );

std::filesystem::path
get_filename( const std::map<std::string, Index> & file_params );

// std::filesystem::path
// get_metadata_filename( const std::map<char, Index> & hyperparams ) {}

template <Weight T>
DataPair<T>
train_split( const ConstRefMat<T> raw_data, const FeatureVecShape & shape,
             const Index k, const Index s, const Index stride = 1 ) {
    Mat<T> data{ raw_data(
        Eigen::seq( Eigen::fix<0>, Eigen::placeholders::last, stride ),
        Eigen::placeholders::all ) };

    const Index max_delay{ std::ranges::max( shape
                                             | std::views::elements<1> ) },
        n{ static_cast<Index>( data.rows() ) },
        d{ static_cast<Index>( shape.size() ) },
        train_size{ n - max_delay - 1 },
        label_size{ n - max_delay - 1 - s * ( k - 1 ) },
        label_offset{ s * ( k - 1 ) };

    Mat<T> train_samples( train_size, d ), train_labels( label_size, d );

    for ( const auto [i, feature_data] : shape | std::views::enumerate ) {
        const auto [data_col, delay] = feature_data;
        const auto offset{ max_delay - delay };

        train_samples.col( i ) =
            data( Eigen::seq( offset, Eigen::placeholders::last - delay - 1 ),
                  data_col );

        train_labels.col( i ) =
            data( Eigen::seq( offset + label_offset + 1,
                              Eigen::placeholders::last - delay ),
                  data_col );
    }

    return std::tuple{ train_samples, train_labels };
}

template <Weight T>
DataPair<T>
test_split( const ConstRefMat<T> raw_data, const FeatureVecShape & shape,
            const Index k, const Index s, const Index stride = 1 ) {
    Mat<T> data{ raw_data(
        Eigen::seq( Eigen::fix<0>, Eigen::placeholders::last, stride ),
        Eigen::placeholders::all ) };

    const Index max_delay{ std::ranges::max( shape
                                             | std::views::elements<0> ) },
        d{ static_cast<Index>( shape.size() ) },
        n{ static_cast<Index>( data.rows() ) }, warmup_offset{ s * ( k - 1 ) },
        test_sz{ n - max_delay - warmup_offset - 1 };

    Mat<T> test_warmup( warmup_offset + 1, d ), test_labels( test_sz, d );

    for ( const auto [i, feature_data] : shape | std::views::enumerate ) {
        const auto [data_col, delay] = feature_data;
        const auto offset{ max_delay - delay };

        test_warmup.col( i ) =
            data( Eigen::seq( offset, offset + warmup_offset ), data_col );


        test_labels.col( i ) =
            data( Eigen::seq( offset + warmup_offset + 1,
                              Eigen::placeholders::last - delay ),
                  data_col );
    }

    return std::tuple{ test_warmup, test_labels };
}

template <Weight T>
std::tuple<DataPair<T>, DataPair<T>>
data_split( const ConstRefMat<T> train_data, const ConstRefMat<T> test_data,
            const FeatureVecShape & shape, const Index k, const Index s,
            const Index stride = 1 ) {
    return { train_split<T>( train_data, shape, k, s, stride ),
             test_split<T>( test_data, shape, k, s, stride ) };
}

template <Weight T>
std::tuple<DataPair<T>, DataPair<T>>
data_split( const ConstRefMat<T> data, const double train_test_ratio,
            const FeatureVecShape & shape, const Index k, const Index s,
            const Index stride = 1 ) {
    const Index train_size{ static_cast<Index>(
        static_cast<double>( data.rows() ) * train_test_ratio ) },
        test_size{ data.rows() - train_size };

    const ConstRefMat<T> train_data{ data.topRows( train_size ) };
    const ConstRefMat<T> test_data{ data.bottomRows( test_size ) };

    return data_split<T>( train_data, test_data, shape, k, s, stride );
}

} // namespace NVAR
