#pragma once

#include "Eigen/Core"
#include "Eigen/Sparse"

#include <boost/random.hpp>
#include <complex>
#include <filesystem>
#include <format>
#include <iterator>
#include <map>
#include <random>
#include <ranges>
#include <regex>
#include <string>
#include <type_traits>

namespace UTIL
{

// Type concepts

// Complex type checker for complex type
template <typename T>
struct is_complex : std::false_type
{};
template <std::floating_point T>
struct is_complex<std::complex<T>> : std::true_type
{};

// Concept for arithmetic types
template <typename T>
concept ArithmeticType = std::is_arithmetic<T>::value;

template <typename T>
concept InStream = std::convertible_to<T, std::istream &>;
template <typename T>
concept OutStream = std::convertible_to<T, std::ostream &>;

template <typename T>
concept Streamable =
    requires( T x, const T y, std::ostream & os, std::istream & is ) {
        { os << y } -> OutStream;
        { is >> x } -> InStream;
    };

// RandomNumberEngine concept
template <typename Engine>
concept RandomNumberEngine =
    requires( Engine e, const typename Engine::result_type seed,
              const unsigned long long n ) {
        requires std::unsigned_integral<typename Engine::result_type>;
        { e.seed( seed ) } -> std::same_as<void>;
        { e.operator()() } -> std::same_as<typename Engine::result_type>;
        { e.discard( n ) } -> std::same_as<void>;
        { e.min() } -> std::same_as<typename Engine::result_type>;
        { e.max() } -> std::same_as<typename Engine::result_type>;
    }
    && std::default_initializable<Engine>
    && std::constructible_from<Engine, const Engine &>
    && std::constructible_from<Engine, typename Engine::result_type>
    && std::equality_comparable<Engine> && Streamable<Engine>;

// RandomNumberDistribution concept
template <typename Dist>
concept RandomNumberDistribution =
    std::constructible_from<Dist> && std::copyable<Dist>
    && ArithmeticType<typename Dist::result_type>
    && std::copyable<typename Dist::param_type>
    && std::equality_comparable<typename Dist::param_type>
    && requires( Dist x, const Dist y, std::ostream & os, std::istream & is ) {
           // Member types:
           typename Dist::result_type;
           typename Dist::param_type;
>>>>>>> stl_random_init

template <typename T>
concept LessThanComparable =
    std::equality_comparable<T> && requires( const T lhs, const T rhs ) {
        { lhs < rhs } -> std::convertible_to<bool>;
        { lhs > rhs } -> std::convertible_to<bool>;
    };

template <typename T>
concept OutStream = std::is_convertible_v<T, std::ostream &>;
template <typename T>
concept InStream = std::is_convertible_v<T, std::istream &>;

template <typename T>
concept Streamable =
    requires( T x, const T y, std::ostream & os, std::istream & is ) {
        { os << y } -> OutStream;
        { is >> x } -> InStream;
    };

// Concepts for boost random distributions etc.
// template <typename Seq, typename Iter>
// concept SeedSeq = requires( Seq s, Iter begin, Iter end ) {
//    typename Iter::value_type;
//    requires sizeof( Iter::value_type ) >= 4;
//    { s.generate( begin, end ) } -> std::same_as<void>;
//};
//
// template <typename Generator>
// concept NumberGenerator = requires( Generator u ) {
//    typename Generator::result_type;
//    std::numeric_limits<typename Generator::result_type>::is_specialized;
//    requires LessThanComparable<typename Generator::result_type>;
//    { u.operator()() } -> std::convertible_to<typename
//    Generator::result_type>;
//};
//
// template <typename Generator, typename ResultType>
// concept UniformRandomNumberGenerator =
//    NumberGenerator<Generator> && requires( const Generator v ) {
//        // requires std::same_as<typename Generator::has_fixed_range, bool>;
//        // requires std::same_as<typename Generator::min_value,
//        //                       typename Generator::max_value>;
//        //{ v.min() } -> std::same_as<typename Generator::min_value>;
//        //{ v.max() } -> std::same_as<typename Generator::max_value>;
//        { v.min() } -> std::convertible_to<ResultType>;
//        { v.max() } -> std::convertible_to<ResultType>;
//    };
//
// template <typename Generator, typename ResultType, typename Integral>
// concept PseudoRandomNumberGenerator =
//    UniformRandomNumberGenerator<Generator, ResultType>
//    && DefaultConstructible<Generator>
//    && std::constructible_from<Generator, Integral> &&
//    std::copyable<Generator>
//    && std::equality_comparable<Generator> && Streamable<Generator>
//    && std::integral<Integral>
//    && requires( Generator u, Integral i, unsigned long long j ) {
//           { u.seed( i ) } -> std::same_as<void>;
//           { u.discard( j ) } -> std::same_as<void>;
//       };
//
// template <std::integral Seed_t>
// using default_generator =
//    boost::random::mersenne_twister<Seed_t, 624, 397, 31, Seed_t{ 0x9908b0df
//    },
//                                    11, 7, 0x9d2c5680, 15, Seed_t{ 0xefc60000
//                                    }, 18, 334, 6425566U>;
//
// template <typename Distribution, typename Generator>
// concept RandomDistribution =
//    NumberGenerator<Distribution> && std::copyable<Distribution>
//    && Streamable<Distribution> && UniformRandomNumberGenerator<Generator>
//    && requires( Distribution u, const Distribution v, Generator e,
//                 std::ostream & os, std::istream & is ) {
//           typename Distribution::input_type;
//           { u.reset() } -> std::same_as<void>;
//           { u( e ) } -> std::same_as<typename Generator::result_type>;
//       };

// Concept for weight types
template <typename T>
concept Weight = std::floating_point<T>; //|| is_complex<T>::value;

// Typedef for all integral types. Same as Eigen::Index.
using Index = std::ptrdiff_t;

// Eigen vector typedefs
template <Weight T, Index N = Eigen::Dynamic>
using Vec = Eigen::Vector<T, N>;
template <Weight T>
using SVec = Eigen::SparseVector<T, Eigen::ColMajor, Index>;
template <Weight T, Index N = Eigen::Dynamic>
using RowVec = Eigen::RowVector<T, N>;
template <Weight T>
using RowSVec = Eigen::SparseVector<T, Eigen::RowMajor, Index>;
template <Weight T, Index N = Eigen::Dynamic>
using RefVec = Eigen::Ref<Vec<T, N>>;
template <Weight T>
using RefSVec = Eigen::Ref<SVec<T>>;
template <Weight T, Index N = Eigen::Dynamic>
using RefRowVec = Eigen::Ref<RowVec<T, N>>;
template <Weight T>
using RefRowSVec = Eigen::Ref<RowSVec<T>>;
template <Weight T, Index N = Eigen::Dynamic>
using ConstRefVec = Eigen::Ref<const Vec<T, N>>;
template <Weight T>
using CosntRefSVec = Eigen::Ref<const SVec<T>>;
template <Weight T, Index N = Eigen::Dynamic>
using ConstRefRowVec = Eigen::Ref<const RowVec<T, N>>;
template <Weight T>
using ConstRefRowSVec = Eigen::Ref<const RowSVec<T>>;

// Eigen matrix typedefs
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
template <Weight T>
using SMat = Eigen::SparseMatrix<T, Eigen::ColMajor, Index>;
template <Weight T>
using RefSMat = Eigen::Ref<SMat<T>>;
template <Weight T>
using ConstRefSMat = Eigen::Ref<const SMat<T>>;

// Return type for train/test split functions
template <Weight T, Index R = -1, Index C = -1>
using DataPair = std::tuple<Mat<T, R, C>, Mat<T, R, C>>;

// Typedef for representing feature vector data with a column & delay
using FeatureVecShape = std::vector<std::tuple<Index, Index>>;

// String version of Eigen::Matrix shape
template <Weight T, Index R = -1, Index C = -1>
std::string
mat_shape_str( const ConstRefMat<T, R, C> m ) {
    return std::format( "({}, {})", m.rows(), m.cols() );
}

template <Weight T>
constexpr inline T
inversion_condition( const ConstRefMat<T> m ) {
    const auto sing_values = m.jacobiSvd().singularValues();
    return sing_values( Eigen::placeholders::last ) / sing_values( 0 );
}

// Implementation from: https://github.com/pconstr/eigen-ridge.git
template <Weight T>
constexpr inline Mat<T>
tikhonov_regularization( const Mat<T> A, const Mat<T> y, const T alpha ) {
    const auto & svd = A.jacobiSvd( Eigen::ComputeFullU | Eigen::ComputeFullV );
    const auto & s = svd.singularValues();
    const auto   r = s.rows();
    const auto & D =
        s.cwiseQuotient( ( s.array().square() + alpha ).matrix() ).asDiagonal();
    const auto factor = svd.matrixV().leftCols( r ) * D
                        * svd.matrixU().transpose().topRows( r );
    return y.transpose() * factor;
}

inline std::map<std::string, Index>
parse_filename( const std::string_view filename ) {
    const std::string re{
        "([\\d]+)_([01])_([-\\d]+)_([\\d+]+)_([01])_([\\d]+)_([\\d]+)\\.csv"
    };
    const std::regex                                     pattern{ re };
    std::match_results<std::string_view::const_iterator> match;
    std::map<std::string, Index>                         result = {
        { "N", -1 },       { "train_test", -1 },          { "n_points", -1 },
        { "seed", -1 },    { "measured_integrated", -1 }, { "no", -1 },
        { "total_no", -1 }
    };
    if ( std::regex_search( filename.cbegin(), filename.cend(), match,
                            pattern ) ) {
        result["N"] = std::stol( match[0].str() );
        result["train_test"] = std::stol( match[1].str() );
        result["n_points"] = std::stol( match[2].str() );
        result["seed"] = std::stol( match[3].str() );
        result["measured_integrated"] = std::stol( match[4].str() );
        result["no"] = std::stol( match[5].str() );
        result["total_no"] = std::stol( match[6].str() );
    }

    return result;
}

inline std::filesystem::path
get_filename( const std::vector<Index> & params ) {
    std::filesystem::path path{};

    switch ( params[1] ) {
    case 0: {
        path += "train_data";
    } break;
    case 1: {
        path += "test_data";
    } break;
    default: {
        path += "forecast_data";
    } break;
    }

    auto params_joined = params | std::views::transform( []( const Index x ) {
                             return std::to_string( x );
                         } )
                         | std::views::join_with( '_' );

    std::string filename{};
    for ( const auto c : params_joined ) { filename += c; }

    return path /= ( filename + ".csv" );
}

inline std::filesystem::path
get_filename( const std::map<std::string, Index> & file_params ) {
    const auto N{ file_params.at( "N" ) };
    const auto train_test{ file_params.at( "train_test" ) };
    const auto n_points{ file_params.at( "n_points" ) };
    const auto seed{ file_params.at( "seed" ) };
    const auto measured_integrated{ file_params.at( "measured_integrated" ) };
    const auto no{ file_params.at( "no" ) };
    const auto total_no{ file_params.at( "total_no" ) };
    return get_filename( std::vector<Index>{
        N, train_test, n_points, seed, measured_integrated, no, total_no } );
}

// std::filesystem::path
// get_metadata_filename( const std::map<char, Index> & hyperparams ) {}

template <Weight T>
constexpr inline DataPair<T>
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
constexpr inline std::tuple<DataPair<T>, DataPair<T>>
data_split( const ConstRefMat<T> train_data, const ConstRefMat<T> test_data,
            const FeatureVecShape & shape, const Index k, const Index s,
            const Index stride = 1 ) {
    return { train_split<T>( train_data, shape, k, s, stride ),
             test_split<T>( test_data, shape, k, s, stride ) };
}

template <Weight T>
constexpr inline std::tuple<DataPair<T>, DataPair<T>>
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

} // namespace UTIL
