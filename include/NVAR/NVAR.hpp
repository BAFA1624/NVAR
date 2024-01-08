#pragma once

#include "CSV/simple_csv.hpp"
#include "NVAR/NVAR_util.hpp"

#include <filesystem>
#include <format>
#include <vector>

namespace NVAR
{

template <UTIL::Weight T, nonlinear_t Nonlin = nonlinear_t::poly,
          UTIL::Solver S = UTIL::L2Solver<T>, bool target_difference = false>
class NVAR
{
    public:
    // Public typedefs
    using weight_type = T;
    using solver_type = S;
    using index_type = UTIL::Index;

    private:
    // Important NVAR state variables
    UTIL::Index  m_d;
    UTIL::Index  m_k;
    UTIL::Index  m_s;
    UTIL::Index  m_p;
    bool         m_use_constant;
    T            m_c;
    UTIL::Mat<T> m_w_out;

    std::vector<UTIL::Index> m_train_targets;

    // Data sizes
    UTIL::Index m_n_training_inst;  // Number of training samples
    UTIL::Index m_n_linear_feat;    // Size of linear feature vector
    UTIL::Index m_n_nonlinear_feat; // Size of nonlinear feature vector
    UTIL::Index m_n_total_feat;     // Size of total feature vector

    // Solver to train weights
    S m_solver;

    // Other internal variables
    // When true, attempts to reconstruct training data from the training total
    // feature vector & the learned weights
    bool m_reconstruct_training;
    // Column titles for output file when m_reconstruct_training = true
    std::vector<std::string> m_col_titles;
    // File to write reconstructed data to
    std::filesystem::path m_reconstruction_path;

    [[nodiscard]] constexpr inline UTIL::Mat<T>
    transform_labels( const UTIL::ConstRefMat<T> & y ) const noexcept;
    [[nodiscard]] constexpr inline UTIL::Mat<T>
    construct_linear_vec( const UTIL::ConstRefMat<T> & samples ) const noexcept;
    [[nodiscard]] constexpr inline UTIL::Mat<T> construct_nonlinear_vec(
        const UTIL::ConstRefMat<T> & linear_features ) const noexcept;
    [[nodiscard]] constexpr inline UTIL::Vec<T> construct_nonlinear_inst(
        const UTIL::ConstRefVec<T> & linear_features ) const noexcept;
    [[nodiscard]] constexpr inline UTIL::Mat<T> construct_total_vec(
        const UTIL::ConstRefMat<T> & linear_features,
        const UTIL::ConstRefMat<T> & nonlinear_features ) const noexcept;
    [[nodiscard]] constexpr inline UTIL::Mat<T>
    construct_total_vec( const UTIL::ConstRefMat<T> & samples ) const noexcept;

    [[nodiscard]] constexpr inline UTIL::Mat<T>
    train_W_out( const UTIL::ConstRefMat<T> & labels,
                 const UTIL::ConstRefMat<T> & total_feature_vec ) noexcept;

    public:
    NVAR( const UTIL::ConstRefMat<T> & samples,
          const UTIL::ConstRefMat<T> & labels, const UTIL::Index d,
          const UTIL::Index k, const UTIL::Index s, const UTIL::Index p,
          const bool use_constant, const T constant = T{ 0. },
          const std::vector<UTIL::Index> & train_targets = {},
          const UTIL::Solver auto          solver = L2Solver<T>( T{ 0.001 } ),
          const bool                       reconstruct_training = false,
          const std::vector<std::string> & reconstruction_titles = {},
          const std::filesystem::path      reconstruction_path = "" ) :
        m_d( d ),
        m_k( k ),
        m_s( s ),
        m_p( p ),
        m_use_constant( use_constant ),
        m_c( constant ),
        m_train_targets( train_targets ),
        m_n_training_inst( samples.rows() - s * ( k - 1 ) ),
        m_n_linear_feat( d * k ),
        m_n_nonlinear_feat( def_nonlinear_size<Nonlin>( d, k, p ) ),
        m_n_total_feat( def_total_size<Nonlin>( d, k, p, use_constant ) ),
        m_solver( solver ),
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
        const UTIL::Mat<T> linear_features{ construct_linear_vec( samples ) };
        const UTIL::Mat<T> nonlinear_features{ construct_nonlinear_vec(
            linear_features ) };
        const UTIL::Mat<T> total_features{ construct_total_vec(
            linear_features, nonlinear_features ) };

        if constexpr ( target_difference ) {
            m_w_out = train_W_out( transform_labels( labels - samples ),
                                   total_features );
        }
        else {
            m_w_out = train_W_out( transform_labels( labels ), total_features );
        }

        if ( m_reconstruct_training ) {
            // Set column titles to use when writing results out
            if ( static_cast<UTIL::Index>( reconstruction_titles.size() )
                 == m_d ) {
                m_col_titles = reconstruction_titles;

                // Add titles for the labels
                for ( const auto & title : reconstruction_titles ) {
                    std::cout << title << "'" << std::endl;
                    m_col_titles.push_back( title + "'" );
                }
            }
            else {
                // Titles for samples
                for ( UTIL::Index i{ 0 }; i < m_d; ++i ) {
                    m_col_titles.push_back( std::to_string( i ) );
                }
                // Titles for labels
                for ( UTIL::Index i{ 0 }; i < m_d; ++i ) {
                    m_col_titles.push_back( std::to_string( i ) + "'" );
                }
            }

            // Attempt to regenerate training data
            UTIL::Mat<T> regenerated{
                ( m_w_out * total_features ).transpose()
            };
            if constexpr ( target_difference ) {
                regenerated += samples;
                // for ( UTIL::Index r{ 0 }; r < samples.rows(); ++r ) {
                //     regenerated.row( r ) += samples.row( r );
                // }
            }

            // Add labels for plotting
            UTIL::Mat<T> reconstruction( regenerated.rows(),
                                         regenerated.cols() + labels.cols() );
            reconstruction << regenerated, labels;

            // Write to file
            const auto write_success{ CSV::SimpleCSV<T>::template write<T>(
                std::filesystem::absolute( m_reconstruction_path ),
                reconstruction, m_col_titles ) };

            if ( !write_success ) {
                std::cerr << std::format(
                    "Unable to write reconstruction data.\n" );
            }
        }
    }

    [[nodiscard]] constexpr inline auto w_out() const noexcept {
        return m_w_out;
    }

    [[nodiscard]] constexpr inline UTIL::Mat<T>
    forecast( const UTIL::ConstRefMat<T> & warmup,
              const UTIL::ConstRefMat<T> & labels ) const noexcept;
};

template <UTIL::Weight T, nonlinear_t Nonlin, UTIL::Solver S,
          bool target_difference>
[[nodiscard]] constexpr inline UTIL::Mat<T>
NVAR<T, Nonlin, S, target_difference>::transform_labels(
    const UTIL::ConstRefMat<T> & y ) const noexcept {
    if ( m_train_targets.size() != 0 ) {
        // Check pass-through indices are valid
        if ( std::ranges::max( m_train_targets ) >= m_d
             || std::ranges::min( m_train_targets ) < 0 ) {
            std::cerr << std::format(
                "Pass-through indices must meet the requirement: {} > {} && 0 "
                "<= "
                "{}.\n",
                m_d, std::ranges::max( m_train_targets ),
                std::ranges::min( m_train_targets ) );
            exit( EXIT_FAILURE );
        }
    }
    else {
        return y;
    }

    return y( Eigen::placeholders::all, m_train_targets );
}

template <UTIL::Weight T, nonlinear_t Nonlin, UTIL::Solver S,
          bool target_difference>
[[nodiscard]] constexpr inline UTIL::Mat<T>
NVAR<T, Nonlin, S, target_difference>::construct_linear_vec(
    const UTIL::ConstRefMat<T> & samples ) const noexcept {
    UTIL::Mat<T> linear_features( m_n_linear_feat, m_n_training_inst );
    for ( UTIL::Index i{ 0 }; i < m_n_training_inst; ++i ) {
        linear_features.col( i ) = construct_x_i<T>( samples, i, m_k, m_s );
    }
    return linear_features;
}

template <UTIL::Weight T, nonlinear_t Nonlin, UTIL::Solver S,
          bool target_difference>
[[nodiscard]] constexpr inline UTIL::Mat<T>
NVAR<T, Nonlin, S, target_difference>::construct_nonlinear_vec(
    const UTIL::ConstRefMat<T> & linear_features ) const noexcept {
    UTIL::Mat<T> nonlinear_features{ UTIL::Mat<T>::Zero( m_n_nonlinear_feat,
                                                         m_n_training_inst ) };
    if constexpr ( Nonlin == nonlinear_t::exp ) {
        nonlinear_features = linear_features.unaryExpr(
            []( const T x ) { return std::exp( x ); } );
    }
    else {
        for ( UTIL::Index i{ 0 }; i < m_n_training_inst; ++i ) {
            nonlinear_features.col( i ) << combinations_with_replacement<T>(
                linear_features.col( i ), m_d * m_k, m_p );
        }
    }
    return nonlinear_features;
}

template <UTIL::Weight T, nonlinear_t Nonlin, UTIL::Solver S,
          bool target_difference>
[[nodiscard]] constexpr inline UTIL::Vec<T>
NVAR<T, Nonlin, S, target_difference>::construct_nonlinear_inst(
    const UTIL::ConstRefVec<T> & linear_feature ) const noexcept {
    UTIL::Vec<T> nonlinear_feature{ UTIL::Vec<T>::Zero( m_n_nonlinear_feat ) };
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

template <UTIL::Weight T, nonlinear_t Nonlin, UTIL::Solver S,
          bool target_difference>
[[nodiscard]] constexpr inline UTIL::Mat<T>
NVAR<T, Nonlin, S, target_difference>::construct_total_vec(
    const UTIL::ConstRefMat<T> & linear_features,
    const UTIL::ConstRefMat<T> & nonlinear_features ) const noexcept {
    UTIL::Mat<T> total_features{ UTIL::Mat<T>::Zero( m_n_total_feat,
                                                     m_n_training_inst ) };
    if ( m_use_constant ) {
        for ( UTIL::Index i{ 0 }; i < m_n_training_inst; ++i ) {
            total_features.col( i ) << linear_features.col( i ),
                nonlinear_features.col( i ), m_c;
        }
    }
    else {
        for ( UTIL::Index i{ 0 }; i < m_n_training_inst; ++i ) {
            total_features.col( i ) << linear_features.col( i ),
                nonlinear_features.col( i );
        }
    }
    return total_features;
}

template <UTIL::Weight T, nonlinear_t Nonlin, UTIL::Solver S,
          bool target_difference>
[[nodiscard]] constexpr inline UTIL::Mat<T>
NVAR<T, Nonlin, S, target_difference>::construct_total_vec(
    const UTIL::ConstRefMat<T> & samples ) const noexcept {
    const auto linear_features{ construct_linear_vec( samples ) };
    const auto nonlinear_features{ construct_nonlinear_vec( linear_features ) };
    return construct_total_vec( linear_features, nonlinear_features );
}

template <UTIL::Weight T, nonlinear_t Nonlin, UTIL::Solver S,
          bool target_difference>
[[nodiscard]] constexpr inline UTIL::Mat<T>
NVAR<T, Nonlin, S, target_difference>::train_W_out(
    const UTIL::ConstRefMat<T> & labels,
    const UTIL::ConstRefMat<T> & total_feature_vec ) noexcept {
    return m_solver.solve( total_feature_vec.transpose(), labels );
}

template <UTIL::Weight T, nonlinear_t Nonlin, UTIL::Solver S,
          bool target_difference>
[[nodiscard]] constexpr inline UTIL::Mat<T>
NVAR<T, Nonlin, S, target_difference>::forecast(
    const UTIL::ConstRefMat<T> & warmup,
    const UTIL::ConstRefMat<T> & labels ) const noexcept {
    UTIL::Mat<T>    result = labels;
    UTIL::RowVec<T> prev_value{ warmup.row( warmup.rows() - 1 ) };
    UTIL::Mat<T>    samples = warmup;

    for ( UTIL::Index i{ 0 }; i < labels.rows(); ++i ) {
        // Construct total feature vector
        UTIL::Vec<T> lin_feat = construct_x_i<T>( samples, m_k, m_s );
        UTIL::Vec<T> nonlin_feat = construct_nonlinear_inst( lin_feat );
        UTIL::Vec<T> total_feat{ UTIL::Vec<T>::Zero(
            def_total_size<Nonlin>( m_d, m_k, m_p, m_use_constant ) ) };
        if ( m_use_constant ) {
            total_feat << lin_feat, nonlin_feat, m_c;
        }
        else {
            total_feat << lin_feat, nonlin_feat;
        }

        // Make next step prediction
        const auto prediction{ m_w_out * total_feat };

        UTIL::Index j{ 0 };
        for ( const auto idx : m_train_targets ) {
            result( i, idx ) = prediction[j++];
            if constexpr ( target_difference ) {
                result( i, idx ) += prev_value[idx];
            }
        }

        // Cycle samples and set new previous values
        samples = cycle_inputs<T>( samples, result.row( i ) );
        prev_value << result.row( i );
    }

    return result;
}

} // namespace NVAR
