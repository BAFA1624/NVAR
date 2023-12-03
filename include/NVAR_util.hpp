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
        for ( Index i{ j + 1 }; i < k; ++i ) {
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
        indices[j]++;
        for ( Index i{ j + 1 }; i < k; ++i ) { indices[i] = indices[i - 1]; }
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
    std::cout << std::format( "ridge_factor: ({}, {})\n", factor.rows(),
                              factor.cols() );
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

using FeatureVecShape = std::vector<std::tuple<Index, Index>>;

template <Weight T>
std::tuple<Mat<T>, Mat<T>>
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

    std::cout << std::format(
        "n: {}, d: {}, train_size: {}, label_size: {}, label_offset: {}\n", n,
        d, train_size, label_size, label_offset );

    Mat<T> train_samples( train_size, d ), train_labels( label_size, d );

    std::cout << std::format( "train_samples: {}, train_labels: {}\n",
                              mat_shape_str<T, -1, -1>( train_samples ),
                              mat_shape_str<T, -1, -1>( train_labels ) );

    for ( const auto [i, feature_data] : shape | std::views::enumerate ) {
        const auto [data_col, delay] = feature_data;
        const auto offset{ max_delay - delay };
        std::cout << std::format( "data_col: {}, delay: {}, offset: {}\n",
                                  data_col, delay, offset );
        std::cout << std::format( "Setting samples {}\n", i );
        std::cout << std::format(
            "train_samples({}): {}\n", i,
            mat_shape_str<T, -1, -1>( train_samples.col( i ) ) );
        std::cout << std::format( "train_samples({}): {} -> {} (range = {})\n",
                                  i, offset, n - 2 - delay,
                                  n - 1 - delay - offset );

        const auto tmp1 =
            data( Eigen::seq( offset, Eigen::placeholders::last - delay - 1 ),
                  data_col );

        std::cout << std::format( "tmp1: {}\n",
                                  mat_shape_str<T, -1, -1>( tmp1 ) );

        train_samples.col( i ) =
            data( Eigen::seq( offset, Eigen::placeholders::last - delay - 1 ),
                  data_col );

        std::cout << std::format( "Done.\nSetting labels {}\n", i );
        std::cout << std::format(
            "train_labels({}): {}\n", i,
            mat_shape_str<T, -1, -1>( train_labels.col( i ) ) );
        std::cout << std::format( "train_labels({}): {} -> {} (range = {})\n",
                                  i, offset + label_offset + 1, n - 1 - delay,
                                  n - 1 - delay - offset - label_offset );

        const auto tmp2 = data( Eigen::seq( offset + label_offset + 1,
                                            Eigen::placeholders::last - delay ),
                                data_col );
        std::cout << std::format( "tmp2: {}\n",
                                  mat_shape_str<T, -1, -1>( tmp2 ) );

        train_labels.col( i ) =
            data( Eigen::seq( offset + label_offset + 1,
                              Eigen::placeholders::last - delay ),
                  data_col );
        std::cout << "Done." << std::endl;
    }

    std::cout << "Finished train_split.\n\n";

    return std::tuple{ train_samples, train_labels };
}

template <Weight T>
std::tuple<Mat<T>, Mat<T>>
test_split( const ConstRefMat<T> raw_data, const FeatureVecShape & shape,
            const Index k, const Index s, const Index stride = 1 ) {
    Mat<T> data{ raw_data(
        Eigen::seq( Eigen::fix<0>, Eigen::placeholders::last, stride ),
        Eigen::placeholders::all ) };

    std::cout << std::format( "data: {}", mat_shape_str<T, -1, -1>( data ) );

    const Index max_delay{ std::ranges::max( shape
                                             | std::views::elements<0> ) },
        d{ static_cast<Index>( shape.size() ) },
        n{ static_cast<Index>( data.rows() ) }, warmup_offset{ s * ( k - 1 ) },
        test_sz{ n - max_delay - warmup_offset - 1 };

    std::cout << std::format(
        "n: {}, d: {}, max_delay: {}, warmup_sz: {}, test_sz: {}\n", n, d,
        max_delay, warmup_offset + 1, test_sz );

    Mat<T> test_warmup( warmup_offset + 1, d ), test_labels( test_sz, d );

    std::cout << std::format( "test_warmup: {}, test_labels: {}\n",
                              mat_shape_str<T, -1, -1>( test_warmup ),
                              mat_shape_str<T, -1, -1>( test_labels ) );

    for ( const auto [i, feature_data] : shape | std::views::enumerate ) {
        const auto [data_col, delay] = feature_data;
        const auto offset{ max_delay - delay };

        std::cout << std::format( "data_col: {}, delay: {}, offset: {}\n",
                                  data_col, delay, offset );
        std::cout << std::format( "Setting warmup {}\n", i );
        std::cout << std::format(
            "test_warmup({}): {}\n", i,
            mat_shape_str<T, -1, -1>( test_warmup.col( i ) ) );
        std::cout << std::format( "test_warmup({}): {} -> {} (range = {})\n", i,
                                  offset, offset + warmup_offset,
                                  warmup_offset + 1 );

        const auto tmp1 =
            data( Eigen::seq( offset, offset + warmup_offset ), data_col );
        std::cout << std::format( "tmp1: {}\n",
                                  mat_shape_str<T, -1, -1>( tmp1 ) );

        test_warmup.col( i ) =
            data( Eigen::seq( offset, offset + warmup_offset ), data_col );

        std::cout << std::format( "Done.\nSetting labels {}\n", i );
        std::cout << std::format(
            "test_labels({}): {}\n", i,
            mat_shape_str<T, -1, -1>( test_labels.col( i ) ) );
        std::cout << std::format( "test_labels({}): {} -> {} (range = {})\n", i,
                                  offset + warmup_offset + 1, n - delay - 1,
                                  n - delay - offset - warmup_offset );

        const auto tmp2 = data( Eigen::seq( offset + warmup_offset + 1,
                                            Eigen::placeholders::last - delay ),
                                data_col );
        std::cout << std::format( "tmp2: {}\n",
                                  mat_shape_str<T, -1, -1>( tmp2 ) );

        test_labels.col( i ) =
            data( Eigen::seq( offset + warmup_offset + 1,
                              Eigen::placeholders::last - delay ),
                  data_col );

        std::cout << "Done." << std::endl;
    }

    std::cout << std::format( "test_split done.\n" );

    return std::tuple{ test_warmup, test_labels };
}

} // namespace NVAR
