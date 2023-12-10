#pragma once

#include "CSV/simple_csv.hpp"
#include "NVAR/NVAR_util.hpp"

#include <filesystem>
#include <format>
#include <vector>

namespace NVAR
{

template <Weight T, nonlinear_t Nonlin = nonlinear_t::poly,
          bool target_difference = false>
class NVAR
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

    // Indices used to generate nonlinear features
    std::vector<std::vector<Index>> m_nonlinear_indices;

    // Data sizes
    Index m_n_training_inst;  // Number of training samples
    Index m_n_linear_feat;    // Size of linear feature vector
    Index m_n_nonlinear_feat; // Size of nonlinear feature vector
    Index m_n_total_feat;     // Size of total feature vector

    // Other internal variables
    // When true, attempts to reconstruct training data from the training total
    // feature vector & the learned weights
    bool m_reconstruct_training;
    // Column titles for output file when m_reconstruct_training = true
    std::vector<std::string> m_col_titles;
    // File to write reconstructed data to
    std::filesystem::path m_reconstruction_path;

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
    NVAR( const ConstRefMat<T> samples, const ConstRefMat<T> labels,
          const Index d, const Index k, const Index s, const Index p,
          const T ridge_param, const bool use_constant,
          const T constant = T{ 0. }, const bool reconstruct_training = false,
          const std::vector<std::string> & reconstruction_titles = {},
          const std::filesystem::path      reconstruction_path = "" ) :
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
        m_n_total_feat( def_total_size<Nonlin>( d, k, p, use_constant ) ),
        m_reconstruct_training( reconstruct_training ),
        m_reconstruction_path( reconstruction_path ) {
        if ( labels.rows() != m_n_training_inst ) {
            std::cerr << std::format(
                "The number of training labels ({}) must equal the number of "
                "training samples ({}), given the values of d ({}), k ({}), & "
                "s ({}).\n",
                labels.rows(), m_n_training_inst, m_d, m_k, m_s );
            exit( 1 );
        }
        const Mat<T> linear_features{ construct_linear_vec( samples ) };
        const Mat<T> nonlinear_features{ construct_nonlinear_vec(
            linear_features ) };
        const Mat<T> total_features{ construct_total_vec(
            linear_features, nonlinear_features ) };

        if constexpr ( target_difference ) {
            m_w_out = ridge_regress( labels - samples, total_features );
        }
        else {
            m_w_out = ridge_regress( labels, total_features );
        }

        if ( m_reconstruct_training ) {
            // Set column titles to use when writing results out
            if ( static_cast<Index>( reconstruction_titles.size() ) == m_d ) {
                m_col_titles = reconstruction_titles;

                // Add titles for the labels
                for ( const auto & title : reconstruction_titles ) {
                    std::cout << title << "'" << std::endl;
                    m_col_titles.push_back( title + "'" );
                }
            }
            else {
                // Titles for samples
                for ( Index i{ 0 }; i < m_d; ++i ) {
                    m_col_titles.push_back( std::to_string( i ) );
                }
                // Titles for labels
                for ( Index i{ 0 }; i < m_d; ++i ) {
                    m_col_titles.push_back( std::to_string( i ) + "'" );
                }
            }

            // Attempt to regenerate training data
            auto regenerated{ ( m_w_out * total_features ).transpose() };
            if constexpr ( target_difference ) {
                for ( Index r{ 0 }; r < samples.rows(); ++r ) {
                    regenerated.row( r ) += samples.row( r );
                }
            }

            // Add labels for plotting
            Mat<T> reconstruction( regenerated.rows(),
                                   regenerated.cols() + labels.cols() );
            reconstruction << regenerated, labels;

            // Write to file
            if ( std::filesystem::directory_entry( m_reconstruction_path )
                     .exists() ) {
                CSV::SimpleCSV::write<T>( m_reconstruction_path, reconstruction,
                                          m_col_titles );
            }
            else {
                std::cerr << std::format(
                    "Unable to write reconstruction data to path: {}\n",
                    m_reconstruction_path.string() );
            }
        }
    }

    [[nodiscard]] constexpr inline auto w_out() const noexcept {
        return m_w_out;
    }

    [[nodiscard]] constexpr inline Mat<T>
    forecast( const ConstRefMat<T> warmup, const ConstRefMat<T> labels,
              const std::vector<Index> & pass_through ) const noexcept;
};

template <Weight T, nonlinear_t Nonlin, bool target_difference>
[[nodiscard]] constexpr inline Mat<T>
NVAR<T, Nonlin, target_difference>::construct_linear_vec(
    const ConstRefMat<T> samples ) const noexcept {
    Mat<T> linear_features( m_n_linear_feat, m_n_training_inst );
    for ( Index i{ 0 }; i < m_n_training_inst; ++i ) {
        linear_features.col( i ) = construct_x_i<T>( samples, i, m_k, m_s );
    }
    return linear_features;
}

template <Weight T, nonlinear_t Nonlin, bool target_difference>
[[nodiscard]] constexpr inline Mat<T>
NVAR<T, Nonlin, target_difference>::construct_nonlinear_vec(
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
                linear_features.col( i ), m_d * m_k, m_p );
        }
    }
    return nonlinear_features;
}

template <Weight T, nonlinear_t Nonlin, bool target_difference>
[[nodiscard]] constexpr inline Vec<T>
NVAR<T, Nonlin, target_difference>::construct_nonlinear_inst(
    const ConstRefVec<T> linear_feature ) const noexcept {
    Vec<T> nonlinear_feature{ Vec<T>::Zero( m_n_nonlinear_feat ) };
    if constexpr ( Nonlin == nonlinear_t::exp ) {
        nonlinear_feature = linear_feature.unaryExpr(
            []( const T x ) { return std::exp( x ); } );
    }
    else {
        nonlinear_feature =
            combinations_with_replacement<T>( linear_feature, m_d * m_k, m_p );
    }
    return nonlinear_feature;
}

template <Weight T, nonlinear_t Nonlin, bool target_difference>
[[nodiscard]] constexpr inline Mat<T>
NVAR<T, Nonlin, target_difference>::construct_total_vec(
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

template <Weight T, nonlinear_t Nonlin, bool target_difference>
[[nodiscard]] constexpr inline Mat<T>
NVAR<T, Nonlin, target_difference>::construct_total_vec(
    const ConstRefMat<T> samples ) const noexcept {
    const auto linear_features{ construct_linear_vec( samples ) };
    const auto nonlinear_features{ construct_nonlinear_vec( linear_features ) };
    return construct_total_vec( linear_features, nonlinear_features );
}

template <Weight T, nonlinear_t Nonlin, bool target_difference>
[[nodiscard]] constexpr inline Mat<T>
NVAR<T, Nonlin, target_difference>::ridge_regress(
    const ConstRefMat<T> labels,
    const ConstRefMat<T> total_feature_vec ) noexcept {
    const auto feature_vec_product{ total_feature_vec
                                    * total_feature_vec.transpose() };
    const auto tikhonov_matrix{
        m_ridge_param * Mat<T>::Identity( m_n_total_feat, m_n_total_feat )
    };
    // L2 regularization adapts linear regression to ill-posed problems
    const auto sum{ feature_vec_product + tikhonov_matrix };
    const auto factor{ sum.completeOrthogonalDecomposition().pseudoInverse() };
    return labels.transpose() * ( factor * total_feature_vec ).transpose();
}

template <Weight T, nonlinear_t Nonlin, bool target_difference>
[[nodiscard]] constexpr inline Mat<T>
NVAR<T, Nonlin, target_difference>::forecast(
    const ConstRefMat<T> warmup, const ConstRefMat<T> labels,
    const std::vector<Index> & pass_through ) const noexcept {
    [[maybe_unused]] const Index max_idx{ *std::max_element(
        pass_through.cbegin(), pass_through.cend() ) };
    [[maybe_unused]] const Index min_idx{ *std::min_element(
        pass_through.cbegin(), pass_through.cend() ) };
    assert( max_idx < warmup.cols() && min_idx >= 0 );

    const Index N{ labels.rows() };
    Mat<T>      result{ Mat<T>::Zero( N, m_d ) };
    RowVec<T>   new_val( 2 );
    RowVec<T>   prev_value{ warmup.row( warmup.rows() - 1 ) };
    Mat<T>      samples = warmup;

    for ( Index i{ 0 }; i < N; ++i ) {
        // Construct total feature vector
        Vec<T> lin_feat = construct_x_i<T>( samples, m_k, m_s );
        Vec<T> nonlin_feat = construct_nonlinear_inst( lin_feat );
        Vec<T> total_feat{ Vec<T>::Zero(
            def_total_size<Nonlin>( m_d, m_k, m_p, m_use_constant ) ) };
        if ( m_use_constant ) {
            total_feat << lin_feat, nonlin_feat, m_c;
        }
        else {
            total_feat << lin_feat, nonlin_feat;
        }

        // Make next step prediction
        new_val = ( m_w_out * total_feat ).transpose();
        if constexpr ( target_difference ) {
            new_val += prev_value;
        }

        // Pass-through any specified true values
        for ( const auto idx : pass_through ) {
            new_val[idx] = labels( i, idx );
        }

        // Assign prediction to result & cycle samples
        result.row( i ) << new_val;
        samples = cycle_inputs<T>( samples, result.row( i ) );
        prev_value << result.row( i );
    }

    return result;
}

} // namespace NVAR
