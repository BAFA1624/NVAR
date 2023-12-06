#pragma once

#include "NVAR_pipe.hpp"
#include "NVAR_util.hpp"
#include "simple_csv.hpp"

#include <filesystem>
#include <format>
#include <vector>

// #define TARGET_DIFFERENCE
// #define ALT_RIDGE
// #define WRITE_FEATURES 10

namespace NVAR
{

template <Weight T, Index N = -1, Index d = -1, Index k = -1, Index s = -1,
          Index p = -1, bool C = true, nonlinear_t Nonlin = nonlinear_t::poly>
class NVAR
{
    private:
    constexpr static inline const Index n_training_inst{ N - s * ( k - 1 ) };
    constexpr static inline const Index n_linear_feat{ d * k };
    constexpr static inline const Index n_nonlinear_feat{
        def_nonlinear_size<d, k, p, Nonlin>()
    };
    constexpr static inline const Index n_feature_param{
        def_total_size<d, k, p, Nonlin, C>()
    };

    // Ridge parameter
    T m_ridge_param;
    // Output weight matrix
    Mat<T, d, def_total_size<d, k, p, Nonlin, C>()> m_w_out;

    // Private functions to construct feature vectors
    [[nodiscard]] constexpr inline Mat<T, d * k, N - s *( k - 1 )>
    construct_linear_feature_vec(
        const RefMat<T, N, d> samples ) const noexcept;
    [[nodiscard]] constexpr inline Mat<T, def_nonlinear_size<d, k, p, Nonlin>(),
                                       N - s *( k - 1 )>
    construct_nonlinear_feature_vec( const RefMat<T, d * k, N - s *( k - 1 )>
                                         lin_components ) const noexcept;
    [[nodiscard]] constexpr inline Mat<T, def_total_size<d, k, p, Nonlin, C>(),
                                       N - s *( k - 1 )>
    construct_total_feature_vec(
        const RefMat<T, d * k, N - s *( k - 1 )> linear_feat,
        const RefMat<T, def_nonlinear_size<d, k, p, Nonlin>(), N - s *( k - 1 )>
                nonlinear_feat,
        const T constant ) const noexcept;
    [[nodiscard]] constexpr inline Mat<T, def_total_size<d, k, p, Nonlin, C>(),
                                       N - s *( k - 1 )>
    construct_total_feature_vec( const RefMat<T, N, d> samples,
                                 const T constant = T{ 0. } ) const noexcept;

    // Ridge regression for m_w_out
    [[nodiscard]] constexpr inline Mat<T, d,
                                       def_total_size<d, k, p, Nonlin, C>()>
    ridge_regress(
        const RefMat<T, d, N - s *( k - 1 )> labels_transposed,
        const RefMat<T, def_total_size<d, k, p, Nonlin, C>(), N - s *( k - 1 )>
            total_feature_vec ) noexcept;

    public:
    constexpr NVAR( const RefMat<T, N, d>                samples,
                    const RefMat<T, N - s *( k - 1 ), d> labels,
                    const T ridge_param, const T constant = T{ 0. } ) :
        m_ridge_param( ridge_param ) {
        // Create feature vectors
        const auto linear_feature_vec{ construct_linear_feature_vec(
            samples ) };
        const auto nonlinear_feature_vec{ construct_nonlinear_feature_vec(
            linear_feature_vec ) };
        const auto total_feature_vec{ construct_total_feature_vec(
            linear_feature_vec, nonlinear_feature_vec, constant ) };

        // Calculate W_out
        m_w_out = ridge_regress( labels.transpose(), total_feature_vec );
    }

    [[nodiscard]] constexpr inline auto w_out() const noexcept {
        return m_w_out;
    }

    [[nodiscard]] constexpr inline Vec<T, d>
    next( const RefMat<T, k * s, d> warmup_samples ) const noexcept;
    template <Index M>
    [[nodiscard]] constexpr inline Mat<T, M, d>
    forecast( const RefMat<T, k * s, d> warmup_samples ) const noexcept;
};

template <Weight T, Index N, Index d, Index k, Index s, Index p, bool C,
          nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Mat<T, d * k, N - s *( k - 1 )>
NVAR<T, N, d, k, s, p, C, Nonlin>::construct_linear_feature_vec(
    const RefMat<T, N, d> samples ) const noexcept {
    Mat<T, n_linear_feat, n_training_inst> linear_features;
    for ( Index i{ 0 }; i < n_training_inst; ++i ) {
        linear_features.col( i ) << construct_x_i<T, d, k, s, N>( samples, i );
    }
    return linear_features;
}

template <Weight T, Index N, Index d, Index k, Index s, Index p, bool C,
          nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Mat<T, def_nonlinear_size<d, k, p, Nonlin>(),
                                   N - s *( k - 1 )>
NVAR<T, N, d, k, s, p, C, Nonlin>::construct_nonlinear_feature_vec(
    const RefMat<T, d * k, N - s *( k - 1 )> lin_components ) const noexcept {
    Mat<T, n_nonlinear_feat, n_training_inst> nonlinear_features;
    for ( Index i{ 0 }; i < n_training_inst; ++i ) {
        nonlinear_features.col( i )
            << combinations_with_replacement<T, d, k, p>(
                   lin_components.col( i ) );
    }
    return nonlinear_features;
}

template <Weight T, Index N, Index d, Index k, Index s, Index p, bool C,
          nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Mat<T, def_total_size<d, k, p, Nonlin, C>(),
                                   N - s *( k - 1 )>
NVAR<T, N, d, k, s, p, C, Nonlin>::construct_total_feature_vec(
    const RefMat<T, d * k, N - s *( k - 1 )> linear_feat,
    const RefMat<T, def_nonlinear_size<d, k, p, Nonlin>(), N - s *( k - 1 )>
            nonlinear_feat,
    const T constant ) const noexcept {
    Mat<T, n_feature_param, n_training_inst> total_features;
    for ( Index i{ 0 }; i < n_training_inst; ++i ) {
        if constexpr ( C ) {
            total_features.col( i ) << linear_feat.col( i ),
                nonlinear_feat.col( i ), constant;
        }
        else {
            total_features.col( i ) << linear_feat.col( i ),
                nonlinear_feat.col( i );
        }
    }
    return total_features;
}

template <Weight T, Index N, Index d, Index k, Index s, Index p, bool C,
          nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Mat<T, def_total_size<d, k, p, Nonlin, C>(),
                                   N - s *( k - 1 )>
NVAR<T, N, d, k, s, p, C, Nonlin>::construct_total_feature_vec(
    const RefMat<T, N, d> samples, const T constant ) const noexcept {
    const Mat<T, n_linear_feat, n_training_inst> linear_feats{
        construct_linear_feature_vec( samples )
    };
    const Mat<T, n_nonlinear_feat, n_training_inst> nonlinear_feats{
        construct_nonlinear_feature_vec( linear_feats )
    };
    return construct_total_feature_vec( linear_feats, nonlinear_feats,
                                        constant );
}

template <Weight T, Index N, Index d, Index k, Index s, Index p, bool C,
          nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Mat<T, d, def_total_size<d, k, p, Nonlin, C>()>
NVAR<T, N, d, k, s, p, C, Nonlin>::ridge_regress(
    const RefMat<T, d, N - s *( k - 1 )> labels_transposed,
    const RefMat<T, def_total_size<d, k, p, Nonlin, C>(), N - s *( k - 1 )>
        total_feature_vec ) noexcept {
    const auto tikhonov_matrix{
        m_ridge_param * Mat<T, n_feature_param, n_feature_param>::Identity()
    };
    const auto feature_vec_product{ total_feature_vec
                                    * total_feature_vec.transpose() };
    const auto factor{ ( feature_vec_product + tikhonov_matrix )
                           .completeOrthogonalDecomposition()
                           .pseudoInverse() };

    return labels_transposed * total_feature_vec.transpose() * factor;
}

template <Weight T, nonlinear_t Nonlin = nonlinear_t::poly>
class NVAR_runtime
{
    private:
    // Important NVAR state variables
    Index  m_d;
    Index  m_k;
    Index  m_s;
    Index  m_p;
    bool   m_use_constant;
    T      m_c;
    T      m_ridge_param;
    Mat<T> m_w_out;

    //
    Index m_n_training_inst;
    Index m_n_linear_feat;
    Index m_n_nonlinear_feat;
    Index m_n_total_feat;

    [[nodiscard]] constexpr inline Mat<T>
    construct_linear_vec( const ConstRefMat<T> samples ) const noexcept;
    [[nodiscard]] constexpr inline Mat<T> construct_nonlinear_vec(
        const ConstRefMat<T> linear_features ) const noexcept;
    [[nodiscard]] constexpr inline Vec<T> construct_nonlinear_inst(
        const ConstRefVec<T> linear_features ) const noexcept;
    [[nodiscard]] constexpr inline Mat<T> construct_total_vec(
        const ConstRefMat<T> linear_features,
        const ConstRefMat<T> nonlinear_features ) const noexcept;
    [[nodiscard]] constexpr inline Mat<T>
    construct_total_vec( const ConstRefMat<T> samples ) const noexcept;

    [[nodiscard]] constexpr inline Mat<T>
    ridge_regress( const ConstRefMat<T> labels,
                   const ConstRefMat<T> total_feature_vec ) noexcept;

    public:
    NVAR_runtime( const ConstRefMat<T> samples, const ConstRefMat<T> labels,
                  const Index d, const Index k, const Index s, const Index p,
                  const T ridge_param, const bool use_constant,
                  const T constant = T{ 0. } ) :
        m_d( d ),
        m_k( k ),
        m_s( s ),
        m_p( p ),
        m_use_constant( use_constant ),
        m_c( constant ),
        m_ridge_param( ridge_param ),
        m_n_training_inst( samples.rows() - s * ( k - 1 ) ),
        m_n_linear_feat( d * k ),
        m_n_nonlinear_feat( def_nonlinear_size<Nonlin>( d, k, p ) ),
        m_n_total_feat( def_total_size<Nonlin>( d, k, p, use_constant ) ) {
        if ( labels.rows() != m_n_training_inst ) {
            std::cerr << std::format(
                "The number of training labels ({}) must equal the number of "
                "training samples ({}), given the values of d ({}), k ({}), & "
                "s ({}).\n",
                labels.rows(), m_n_training_inst, m_d, m_k, m_s );
            exit( 1 );
        }
        // std::cout << "Constructing linear features.\n";
        const Mat<T> linear_features{ construct_linear_vec( samples ) };
        // std::cout << linear_features.leftCols( 10 ) << "\n";
        //  std::cout << "Constructing nonlinear features.\n";
        const Mat<T> nonlinear_features{ construct_nonlinear_vec(
            linear_features ) };
        // std::cout << nonlinear_features.leftCols( 10 ) << "\n";
        //  std::cout << "Constructing total features.\n";
        const Mat<T> total_features{ construct_total_vec(
            linear_features, nonlinear_features ) };

        std::cout << std::format( "m_n_training_inst: {}, total_features: {}\n",
                                  m_n_training_inst,
                                  mat_shape_str<T, -1, -1>( total_features ) );
        // std::cout << total_features.leftCols( 10 ) << "\n";
        // std::cout << "Performing ridge regression.\n";
#ifdef TARGET_DIFFERENCE
        m_w_out = ridge_regress( labels - samples, total_features );
#else
        m_w_out = ridge_regress( labels, total_features );
#endif
        // std::cout << "m_w_out:\n" << m_w_out.transpose() << std::endl;
#ifndef HH_MODEL
    #ifdef FORECAST
        const std::filesystem::path    path{ "../data/forecast_data/tmp.csv" };
        const std::vector<std::string> col_titles{ "I", "V", "I'", "V'" };
    #endif
    #ifdef CUSTOM_FEATURES
        const std::filesystem::path    path{ "../data/forecast_data/tmp.csv" };
        const std::vector<std::string> col_titles{ "V_(n)",    "V_(n-1)",
                                                   "I_(n)",    "V_(n)'",
                                                   "V_(n-1)'", "I_(n)'" };
    #endif
    #ifdef DOUBLESCROLL
        const std::filesystem::path path{
            "../data/forecast_data/doublescroll_reconstruct.csv"
        };
        const std::vector<std::string> col_titles{ "v1", "v2", "I" };
    #endif
    #ifdef HH_MODEL
        const std::filesystem::path    path{ "../data/forecast_data/tmp.csv" };
        const std::vector<std::string> col_titles{ "Vmembrane", "Istim" };
    #endif
        // std::cout << std::format(
        //     "m_w_out: ({}, {}), total_features: ({}, {})\n", m_w_out.rows(),
        //     m_w_out.cols(), total_features.rows(), total_features.cols() );
    #ifndef TARGET_DIFFERENCE
        Mat<T> reproduced = ( m_w_out * total_features ).transpose();
    #else
        Mat<T> reproduced_differences =
            ( m_w_out * total_features ).transpose();
        Mat<T> reproduced = samples.bottomRows( reproduced_differences.rows() );
        std::cout << std::format( "samples: ({}, {})\n", samples.rows(),
                                  samples.cols() );
        std::cout << std::format( "reproduced_differences: ({}, {})\n",
                                  reproduced_differences.rows(),
                                  reproduced_differences.cols() );
        for ( Index c{ 0 }; c < reproduced_differences.rows(); ++c ) {
            reproduced.row( c ) += reproduced_differences.row( c );
        }
    #endif
        std::cout << std::format( "reproduced: ({}, {})\n", reproduced.rows(),
                                  reproduced.cols() );

        Mat<T> reconstructed( reproduced.rows(), reproduced.cols() * 2 );
        reconstructed << reproduced, labels;
        SimpleCSV::write<T>( path, reconstructed, col_titles );
#endif
    }

    [[nodiscard]] constexpr inline auto w_out() const noexcept {
        return m_w_out;
    }

    [[nodiscard]] constexpr inline Mat<T>
    forecast( const ConstRefMat<T> warmup, const ConstRefMat<T> labels,
              const std::vector<Index> & pass_through ) const noexcept;
};

template <Weight T, nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Mat<T>
NVAR_runtime<T, Nonlin>::construct_linear_vec(
    const ConstRefMat<T> samples ) const noexcept {
    Mat<T> linear_features( m_n_linear_feat, m_n_training_inst );
    for ( Index i{ 0 }; i < m_n_training_inst; ++i ) {
        linear_features.col( i ) = construct_x_i<T>( samples, i, m_k, m_s );
    }
    return linear_features;
}

template <Weight T, nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Mat<T>
NVAR_runtime<T, Nonlin>::construct_nonlinear_vec(
    const ConstRefMat<T> linear_features ) const noexcept {
    Mat<T> nonlinear_features{ Mat<T>::Zero( m_n_nonlinear_feat,
                                             m_n_training_inst ) };
    if constexpr ( Nonlin == nonlinear_t::exp ) {
        nonlinear_features = linear_features.unaryExpr(
            []( const T x ) { return std::exp( x ); } );
    }
    else {
        for ( Index i{ 0 }; i < m_n_training_inst; ++i ) {
            nonlinear_features.col( i ) << combinations_with_replacement<T>(
                linear_features.col( i ), m_d, m_k, m_p );
        }
    }
    return nonlinear_features;
}

template <Weight T, nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Vec<T>
NVAR_runtime<T, Nonlin>::construct_nonlinear_inst(
    const ConstRefVec<T> linear_feature ) const noexcept {
    Vec<T> nonlinear_feature{ Vec<T>::Zero( m_n_nonlinear_feat ) };
    if constexpr ( Nonlin == nonlinear_t::exp ) {
        nonlinear_feature = linear_feature.unaryExpr(
            []( const T x ) { return std::exp( x ); } );
    }
    else {
        nonlinear_feature =
            combinations_with_replacement<T>( linear_feature, m_d, m_k, m_p );
    }
    return nonlinear_feature;
}

template <Weight T, nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Mat<T>
NVAR_runtime<T, Nonlin>::construct_total_vec(
    const ConstRefMat<T> linear_features,
    const ConstRefMat<T> nonlinear_features ) const noexcept {
    Mat<T> total_features{ Mat<T>::Zero( m_n_total_feat, m_n_training_inst ) };
    if ( m_use_constant ) {
        for ( Index i{ 0 }; i < m_n_training_inst; ++i ) {
            total_features.col( i ) << linear_features.col( i ),
                nonlinear_features.col( i ), m_c;
        }
    }
    else {
        for ( Index i{ 0 }; i < m_n_training_inst; ++i ) {
            total_features.col( i ) << linear_features.col( i ),
                nonlinear_features.col( i );
        }
    }
    return total_features;
}

template <Weight T, nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Mat<T>
NVAR_runtime<T, Nonlin>::construct_total_vec(
    const ConstRefMat<T> samples ) const noexcept {
    const auto linear_features{ construct_linear_vec( samples ) };
    const auto nonlinear_features{ construct_nonlinear_vec( linear_features ) };
    return construct_total_vec( linear_features, nonlinear_features );
}

template <Weight T, nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Mat<T>
NVAR_runtime<T, Nonlin>::ridge_regress(
    const ConstRefMat<T> labels,
    const ConstRefMat<T> total_feature_vec ) noexcept {
    const auto feature_vec_product{ total_feature_vec
                                    * total_feature_vec.transpose() };
    std::cout << "feature vec det: " << feature_vec_product.determinant()
              << std::endl;
    const auto tikhonov_matrix{
        m_ridge_param * Mat<T>::Identity( m_n_total_feat, m_n_total_feat )
    };
    std::cout << "tikhonov_matrix det: " << tikhonov_matrix.determinant()
              << std::endl;

    std::cout << std::format( "feature_vec: ({}, {})\n",
                              feature_vec_product.rows(),
                              feature_vec_product.cols() );
    std::cout << std::format( "tikhonov_matrix: ({}, {})\n",
                              tikhonov_matrix.rows(), tikhonov_matrix.cols() );

    // L2 regularization adapts linear regression to ill-posed problems
    const auto sum{ feature_vec_product + tikhonov_matrix };
    const auto factor{ sum.completeOrthogonalDecomposition().pseudoInverse() };

    std::cout << std::format( "labels: {}, factor: {}, total_feature_vec: {}\n",
                              mat_shape_str<T, -1, -1>( labels ),
                              mat_shape_str<T, -1, -1>( factor ),
                              mat_shape_str<T, -1, -1>( total_feature_vec ) );

    const auto result{ labels.transpose()
                       * ( factor * total_feature_vec ).transpose() };
    return result;
}

template <Weight T, nonlinear_t Nonlin>
[[nodiscard]] constexpr inline Mat<T>
NVAR_runtime<T, Nonlin>::forecast(
    const ConstRefMat<T> warmup, const ConstRefMat<T> labels,
    const std::vector<Index> & pass_through ) const noexcept {
    [[maybe_unused]] const Index max_idx{ *std::max_element(
        pass_through.cbegin(), pass_through.cend() ) };
    [[maybe_unused]] const Index min_idx{ *std::min_element(
        pass_through.cbegin(), pass_through.cend() ) };
    assert( max_idx < warmup.cols() && min_idx >= 0 );

    const bool show{ true };
    std::cout << std::format( "Starting forecast.\n" );

    const Index N{ labels.rows() };
    Mat<T>      result{ Mat<T>::Zero( N, m_d ) };
    std::cout << std::format( "result: {}\n",
                              mat_shape_str<T, -1, -1>( result ) );
    RowVec<T> new_val( 2 );
    std::cout << std::format( "new_val: {}\n",
                              mat_shape_str<T, -1, -1>( new_val ) );
#ifdef TARGET_DIFFERENCE
    Mat<T> prev_value{ warmup.row( warmup.rows() - 1 ) };
#endif
    Mat<T> samples = warmup;
    std::cout << std::format( "samples: {}, expected: ({}, {})\n",
                              mat_shape_str<T, -1, -1>( samples ),
                              m_s * ( m_k - 1 ), m_d );
#ifdef WRITE_FEATURES
    Mat<T> forecast_features( m_n_total_feat, N );
#endif

    for ( Index i{ 0 }; i < N; ++i ) {
        Vec<T> lin_feat = construct_x_i<T>( samples, m_k, m_s );
        if ( show ) {
            std::cout << std::format( "lin_feat: {}\n",
                                      mat_shape_str<T, -1, -1>( lin_feat ) );
        }
        Vec<T> nonlin_feat = construct_nonlinear_inst( lin_feat );
        if ( show ) {
            std::cout << std::format( "nonlin_feat: {}\n",
                                      mat_shape_str<T, -1, -1>( nonlin_feat ) );
        }
        Vec<T> total_feat{ Vec<T>::Zero(
            def_total_size<Nonlin>( m_d, m_k, m_p, m_use_constant ) ) };
        if ( show ) {
            std::cout << std::format( "total_feat: {}\n",
                                      mat_shape_str<T, -1, -1>( total_feat ) );
        }
        if ( m_use_constant ) {
            total_feat << lin_feat, nonlin_feat, m_c;
        }
        else {
            total_feat << lin_feat, nonlin_feat;
        }
#ifdef WRITE_FEATURES
        forecast_features.col( i ) = total_feat;
#endif
#ifdef TARGET_DIFFERENCE
        new_val = prev_value + ( m_w_out * total_feat ).transpose();
#else
        if ( show ) {
            std::cout << std::format(
                "(m_w_out * total_feat).transpose(): {}\n",
                mat_shape_str<T, -1, -1>(
                    ( m_w_out * total_feat ).transpose() ) );
        }
        new_val = ( m_w_out * total_feat ).transpose();
#endif
        for ( const auto idx : pass_through ) {
            if ( show ) {
                std::cout << "Replacing " << idx << " ";
            }
            new_val[idx] = labels( i, idx );
        }
        if ( show ) {
            std::cout << std::endl;
            std::cout << std::format(
                "result.row({}): {}\n", i,
                mat_shape_str<T, -1, -1>( result.row( i ) ) );
        }
        result.row( i ) << new_val;
        if ( show ) {
            std::cout << std::format( "cycle_inputs: {}\n",
                                      mat_shape_str<T, -1, -1>( cycle_inputs<T>(
                                          samples, result.row( i ) ) ) );
            std::cout << std::format( "samples: {}\n",
                                      mat_shape_str<T, -1, -1>( samples ) );
        }
        samples = cycle_inputs<T>( samples, result.row( i ) );
#ifdef TARGET_DIFFERENCE
        prev_value << result.row( i );
#endif
    }
#ifdef WRITE_FEATURES
    const std::filesystem::path feature_path = {
        "../data/forecast_data/features1.csv"
    };
    SimpleCSV::write<T>( feature_path,
                         forecast_features.leftCols( WRITE_FEATURES ), {} );
#endif


    std::cout << std::format( "labels: {}, forecast result: {}\n",
                              mat_shape_str<T, -1, -1>( labels ),
                              mat_shape_str<T, -1, -1>( result ) );

    return result;
}

} // namespace NVAR
