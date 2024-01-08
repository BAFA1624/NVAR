#pragma once

#include "ESN_util.hpp"
// #include "nlohmann/json.hpp"
#include "util/common.hpp"

namespace ESN
{

template <UTIL::Weight T, input_t W_in_init = input_t::default_input,
          adjacency_t              A_init = adjacency_t::dense,
          feature_t                Feature_init = feature_t::default_feature,
          UTIL::Solver             S = UTIL::L2Solver<T>,
          UTIL::RandomNumberEngine Generator = std::mersenne_twister_engine<
              unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,
              0x9d2c5680, 15, 0xefc60000, 18, 1812433253>,
          bool                  target_difference = false,
          Eigen::StorageOptions _Options = Eigen::RowMajor,
          std::signed_integral  _StorageIndex = UTIL::Index>
class ESN
{
    public:
    // typedefs
    using weight_type = T;
    using solver_type = S;
    using gen_type = Generator;
    using index_type = _StorageIndex;

    private:
    using Seed_t = typename Generator::result_type;
    // Compile time check whether to use dense/sparse input weights & adjacency
    // matrix
    constexpr static bool m_dense_input =
        static_cast<bool>( W_in_init & input_t::dense );
    constexpr static bool m_dense_adjacency =
        static_cast<bool>( A_init & adjacency_t::dense );
    constexpr static bool m_feature_bias =
        static_cast<bool>( Feature_init & feature_t::bias );
    constexpr static bool m_feature_linear =
        static_cast<bool>( Feature_init & feature_t::linear );

    // State variables for the ESN
    UTIL::Index m_d;
    UTIL::Index m_n_node;
    T           m_leak;
    T           m_sparsity;
    T           m_spectral_radius;
    Seed_t      m_seed;
    UTIL::Index m_n_warmup;
    T           m_bias;
    T           m_input_scale;
    bool        m_train_complete;

    // Input weights for the ESN
    UTIL::SMat<T, _Options, _StorageIndex> m_w_in;
    UTIL::Mat<T>                           m_w_in_dense;
    // Adjacency matrix for the reservoir
    UTIL::SMat<T, _Options, _StorageIndex> m_adjacency;
    UTIL::Mat<T>                           m_adjacency_dense;
    // Reservoir state at previous time step
    UTIL::Vec<T> m_reservoir;
    // Linear input at previous time step
    UTIL::Vec<T> m_xi;
    // Linear input * input weights at previous time step
    UTIL::Vec<T> m_wi_xi;
    // Output weights
    UTIL::Mat<T> m_w_out;

    // Storage for which columns in the labels should be the training targets
    std::vector<UTIL::Index> m_train_targets;

    // Common random number generator for all random processes
    Generator m_gen;

    // Solver class to determin W_out
    S m_solver;

    // Activation function for network
    std::function<T( const T )> m_activation;
    // Initialisation function for the input weights
    std::function<T( const T, Generator & )> m_W_in_func;
    // Initialisation function for the adjacency weights
    std::function<T( const T, Generator & )> m_adjacency_func;

    // Expected size of state feature vector
    [[nodiscard]] constexpr inline UTIL::Index feature_size() const noexcept {
        return ( m_feature_bias ? 1 : 0 ) + ( m_feature_linear ? m_d : 0 )
               + m_n_node;
    }

    [[nodiscard]] constexpr inline UTIL::SMat<T, _Options, _StorageIndex>
    init_W_in() noexcept; // Implemented, called by constructor
    [[nodiscard]] constexpr inline UTIL::Mat<T>
    init_W_in_dense() noexcept; // Implemented, called by constructor
    [[nodiscard]] constexpr inline UTIL::SMat<T, _Options, _StorageIndex>
    init_adjacency(
        const UTIL::Index max_iter = 10000,
        const T tol = 1E-10 ) noexcept; // Implemented, called by constructor
    [[nodiscard]] constexpr inline UTIL::Mat<T> init_adjacency_dense(
        const UTIL::Index max_iter = 10000,
        const T tol = 1E-10 ) noexcept; // Implemented, called by constructor
    [[nodiscard]] constexpr inline UTIL::Mat<T>
    wi_xi( const UTIL::ConstRefMat<T> & xi ) const noexcept;
    [[nodiscard]] constexpr inline UTIL::Mat<T>
    A_R( const UTIL::ConstRefMat<T> & R ) const noexcept;
    [[nodiscard]] constexpr inline UTIL::Vec<T>
    R_next( const UTIL::ConstRefVec<T> & R,
            const UTIL::ConstRefVec<T> & Wi_Xi ) const noexcept; // Implemented
    [[nodiscard]] constexpr inline UTIL::Mat<T>
    construct_reservoir_states( const UTIL::ConstRefMat<T> & X ) const noexcept;
    [[nodiscard]] constexpr inline UTIL::ConstRefMat<T>
    transform_labels( const UTIL::ConstRefMat<T> & y ) const noexcept;
    [[nodiscard]] constexpr inline UTIL::Vec<T>
    construct_feature( [[maybe_unused]] const UTIL::ConstRefVec<T> & Xi,
                       const UTIL::ConstRefVec<T> &                  Ri )
        const noexcept; // Implemented, called by train & forecast
    [[nodiscard]] constexpr inline UTIL::Mat<T>
    construct_all_features( [[maybe_unused]] const UTIL::ConstRefMat<T> & X,
                            const UTIL::ConstRefMat<T> &                  R )
        const noexcept; // Implemented, called by train & forecast

    public:
    ESN(
        const UTIL::Index d, const UTIL::Index n_node, const T leak,
        const T sparsity, const T spectral_radius, const Seed_t seed = 0,
        const UTIL::Index n_warmup = 100, const T bias = T{ 1. },
        const T                             input_scale = T{ 1. },
        const std::vector<UTIL::Index> &    train_targets = {},
        const std::function<T( const T )> & activation =
            []( const T x ) { return std::tanh( x ); },
        const S solver = L2Solver<T>( T{ 1E-3 } ),
        const std::function<T( const T, Generator & )> & W_in_func =
            []( [[maybe_unused]] const T x, [[maybe_unused]] Generator & gen ) {
                static auto dist{ std::uniform_real_distribution<T>( -1.,
                                                                     1. ) };
                return dist( gen );
            },
        const std::function<T( const T, Generator & )> & adjacency_func =
            []( [[maybe_unused]] const T x, [[maybe_unused]] Generator & gen ) {
                static auto dist{ std::uniform_real_distribution<T>( -1.,
                                                                     1. ) };
                return dist( gen );
            } );

    // Getters
    [[nodiscard]] constexpr inline UTIL::ConstRefSMat<T, _Options,
                                                      _StorageIndex>
    W_in() const noexcept {
        static_assert( !m_dense_input,
                       "Sparse input weights are uninitialised when "
                       "input_t::dense is defined.\n" );
        return m_w_in;
    }
    [[nodiscard]] constexpr inline UTIL::ConstRefMat<T>
    W_in_dense() const noexcept {
        static_assert( m_dense_input,
                       "Dense input weights are uninitialised when "
                       "input_t::sparse is defined.\n" );
        return m_w_in_dense;
    }
    [[nodiscard]] constexpr inline UTIL::ConstRefMat<T> W_out() const noexcept {
        return m_w_out;
    }
    [[nodiscard]] constexpr inline UTIL::ConstRefSMat<T, _Options,
                                                      _StorageIndex>
    A() const noexcept {
        static_assert( !m_dense_adjacency,
                       "Sparse adjacency weights are uninitialised when "
                       "adjacency_t::dense is defined.\n" );
        return m_adjacency;
    }
    [[nodiscard]] constexpr inline UTIL::ConstRefMat<T>
    A_dense() const noexcept {
        static_assert( m_dense_adjacency,
                       "Dense adjacency weights are uninitialised when "
                       "adjacency_t::sparse is defined.\n" );
        return m_adjacency_dense;
    }

    constexpr inline void train( const UTIL::ConstRefMat<T> & X,
                                 const UTIL::ConstRefMat<T> & y ) noexcept;
    [[nodiscard]] constexpr inline UTIL::Mat<T>
    forecast( const UTIL::Index            n,
              const UTIL::ConstRefMat<T> & warmup = {} ) noexcept;
    [[nodiscard]] constexpr inline UTIL::Mat<T>
    forecast( const UTIL::ConstRefMat<T> & labels ) noexcept;
};

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::
    ESN( const UTIL::Index d, const UTIL::Index n_node, const T leak,
         const T sparsity, const T spectral_radius, const Seed_t seed,
         const UTIL::Index n_warmup, const T bias, const T input_scale,
         const std::vector<UTIL::Index> &    train_targets,
         const std::function<T( const T )> & activation, S solver,
         const std::function<T( const T, Generator & )> & W_in_func,
         const std::function<T( const T, Generator & )> & adjacency_func ) :
    m_d( d ),
    m_n_node( n_node ),
    m_leak( leak ),
    m_sparsity( sparsity ),
    m_spectral_radius( spectral_radius ),
    m_seed( seed ),
    m_n_warmup( n_warmup ),
    m_bias( bias ),
    m_input_scale( input_scale ),
    m_train_complete( false ),
    m_reservoir( UTIL::Vec<T>( n_node ) ),
    m_train_targets( train_targets ),
    m_gen( seed ),
    m_solver( solver ),
    m_activation( activation ),
    m_W_in_func( W_in_func ),
    m_adjacency_func( adjacency_func ) {
    m_gen.discard( static_cast<unsigned long long>( m_gen() ) );

    // Initialise input weights
    // Check that one of input_t::split or input_t::homogeneous is
    // defined, but not both.
    static_assert( static_cast<bool>( ( W_in_init & input_t::split )
                                      ^ ( W_in_init & input_t::homogeneous ) ),
                   "Define either input_t::split OR input_t::homogeneous.\n" );

    // Check that one of input_t::sparse or input_t::dense is defined,
    // but not both.
    static_assert( static_cast<bool>( ( W_in_init & input_t::sparse )
                                      ^ ( W_in_init & input_t::dense ) ),
                   "Define either input_t::sparse OR input_t::dense.\n" );

    // Check that one of adjacency_t::sparse or adjacency_t::dense is defined,
    // but not both.
    static_assert(
        static_cast<bool>( ( A_init & adjacency_t::sparse )
                           ^ ( A_init & adjacency_t::dense ) ),
        "Define either adjacency_t::sparse OR adjacency_t::dense.\n" );

    // Check that at least feature_t::reservoir is defined.
    static_assert( static_cast<bool>( Feature_init & feature_t::reservoir ),
                   "feature_t::reservoir MUST be defined, it can be combined "
                   "with feature_t::constant AND/OR feature_t::linear to "
                   "construct different feature vectors.\n" );

    // Initialise input weights
    std::cout << "Input weights..." << std::endl;
    if constexpr ( !m_dense_input ) {
        m_w_in = init_W_in();
    }
    else {
        m_w_in_dense = init_W_in_dense();
    }

    // Initialise adjacency matrix
    std::cout << "Adjacency..." << std::endl;
    if constexpr ( !m_dense_adjacency ) {
        m_adjacency = init_adjacency();
    }
    else {
        m_adjacency_dense = init_adjacency_dense();
    }
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::SMat<T, _Options, _StorageIndex>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::init_W_in() noexcept {
    constexpr auto split{ static_cast<bool>( W_in_init & input_t::split ) };
    // Generate input weights w/ extra column for bias
    return generate_sparse<T, split, Generator>(
        m_n_node, m_d + ( m_feature_bias ? 1 : 0 ), m_sparsity, m_gen,
        m_W_in_func );
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::Mat<T>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::init_W_in_dense() noexcept {
    constexpr bool split{ static_cast<bool>( W_in_init & input_t::split ) };

    constexpr UTIL::Index bias_sz{ m_feature_bias ? 1 : 0 };

    UTIL::Mat<T> result( m_n_node, m_d + bias_sz );

    const auto f = [this]( const T x ) { return m_W_in_func( x, m_gen ); };
    if constexpr ( split ) {
        std::vector<UTIL::Index> sizes(
            static_cast<std::size_t>( m_d + bias_sz ),
            m_n_node / ( m_d + bias_sz ) );
        for ( std::size_t i{ 0 };
              i < static_cast<std::size_t>( m_n_node % ( m_d + bias_sz ) );
              ++i ) {
            sizes[i]++;
        }

        UTIL::Index offset{ 0 };
        for ( const auto [i, n_rows] : sizes | std::views::enumerate ) {
            result( Eigen::seq( offset, offset + n_rows - 1 ),
                    static_cast<UTIL::Index>( i ) ) =
                result( Eigen::seq( offset, offset + n_rows - 1 ),
                        static_cast<UTIL::Index>( i ) )
                    .unaryExpr( f );

            offset += n_rows;
        }
    }
    else {
        result = result.unaryExpr( f );
    }

    return result;
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::SMat<T, _Options, _StorageIndex>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::init_adjacency( const UTIL::Index max_iter,
                                              const T           tol ) noexcept {
    UTIL::SMat<T, _Options, _StorageIndex> A =
        generate_sparse<T, false, Generator, _Options, _StorageIndex>(
            m_n_node, m_n_node, m_sparsity, m_gen, m_adjacency_func );

    const UTIL::Vec<std::complex<T>> eigenvalues{
        compute_n_eigenvals<T, _Options, _StorageIndex>(
            A, 1, 10, Spectra::SortRule::LargestMagn, max_iter, tol,
            Spectra::SortRule::LargestMagn )
    };
    const T scale_factor{ m_spectral_radius / std::abs<T>( eigenvalues[0] ) };

    return ( A * scale_factor );
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::Mat<T>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::init_adjacency_dense( const UTIL::Index max_iter,
                                                    const T tol ) noexcept {
    // Generate dense matrix according to specified adjacency function
    // Wrapper function to pass to unaryExpr
    const auto adj_func{ [this]( const T x ) {
        return m_adjacency_func( x, m_gen );
    } };

    UTIL::Mat<T> A =
        UTIL::Mat<T>::Zero( m_n_node, m_n_node ).unaryExpr( adj_func );

    const UTIL::Vec<std::complex<T>> eigenvalues{ compute_n_eigenvals_dense<T>(
        A, 1, 10, Spectra::SortRule::LargestMagn, max_iter, tol,
        Spectra::SortRule::LargestMagn ) };
    const T scale_factor{ m_spectral_radius / std::abs<T>( eigenvalues[0] ) };

    return ( A * scale_factor );
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::ConstRefMat<T>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::transform_labels( const UTIL::ConstRefMat<T> & y )
    const noexcept {
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

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::Mat<T>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::wi_xi( const UTIL::ConstRefMat<T> & xi )
    const noexcept {
    if ( xi.rows() != m_d ) {
        std::cerr << std::format(
            "Rows of input to ESN::wi_xi ({}) must match number of dimensions "
            "({}).\n",
            xi.rows(), m_d );
        exit( EXIT_FAILURE );
    }

    constexpr UTIL::Index bias_sz{ m_feature_bias ? 1 : 0 };

    UTIL::Mat<T> input{ UTIL::Mat<T>::Constant( m_d + bias_sz, xi.cols(),
                                                m_bias ) };
    input.bottomRows( m_d ) = xi;

    if constexpr ( m_dense_input ) {
        return m_w_in_dense * input;
    }
    else {
        return m_w_in * input;
    }
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::Mat<T>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::A_R( const UTIL::ConstRefMat<T> & R )
    const noexcept {
    if constexpr ( m_dense_adjacency ) {
        return m_adjacency_dense * R;
    }
    else {
        return m_adjacency * R;
    }
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::Vec<T>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::R_next( const UTIL::ConstRefVec<T> & R,
                                      const UTIL::ConstRefVec<T> & Wi_Xi )
    const noexcept {
    return ( 1 - m_leak ) * R
           + m_leak * ( Wi_Xi + A_R( R ) ).unaryExpr( m_activation );
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::Mat<T>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options,
    _StorageIndex>::construct_reservoir_states( const UTIL::ConstRefMat<T> & X )
    const noexcept {
    // Initialise storage for reservoir states
    UTIL::Mat<T> reservoir_states{ UTIL::Mat<T>::Zero( m_n_node,
                                                       X.cols() + 1 ) };

    // Calculate each data input multiplied with input weights
    UTIL::Mat<T> W_X = wi_xi( X );

    // Calculate each reservoir state
    for ( UTIL::Index col{ 1 }; col < X.cols() + 1; ++col ) {
        reservoir_states.col( col ) =
            R_next( reservoir_states.col( col - 1 ), W_X.col( col - 1 ) );
    }

    // Return all reservoir states, except initial (all zeros)
    return reservoir_states.rightCols( X.cols() );
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::Vec<T>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::
    construct_feature( [[maybe_unused]] const UTIL::ConstRefVec<T> & Xi,
                       const UTIL::ConstRefVec<T> & Ri ) const noexcept {
    UTIL::Vec<T> feature( feature_size() );

    UTIL::Index offset{ 0 };

    // If necessary, add bias to feature vector
    if constexpr ( m_feature_bias ) {
        feature( 0 ) = T{ 1. };
        offset++;
    }

    // If necessary, add linear component to feature vector
    if constexpr ( m_feature_linear ) {
        feature( Eigen::seq( offset, offset + m_d - 1 ) ) = Xi;
        offset += m_d;
    }

    // Add reservoir states to feature vector
    feature( Eigen::seq( offset, feature_size() - 1 ) ) = Ri;

    return feature;
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::Mat<T>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options,
    _StorageIndex>::construct_all_features( const UTIL::ConstRefMat<T> & X,
                                            const UTIL::ConstRefMat<T> & R )
    const noexcept {
    UTIL::Mat<T> features( feature_size(), X.cols() );
    std::cout << std::format( "Feature: {}\n",
                              UTIL::mat_shape_str<T, -1, -1>( features ) );
    UTIL::Index offset{ 0 };

    // If necessary, add bias to feature matrix
    if constexpr ( m_feature_bias ) {
        std::cout << "Adding bias at 0th index." << std::endl;
        features( 0, Eigen::placeholders::all ) =
            UTIL::RowVec<T>::Ones( features.cols() );
        offset++;
    }

    // If necessary, add linear component to feature matrix
    if constexpr ( m_feature_linear ) {
        std::cout << "Adding linear features from " << offset << " - "
                  << offset + m_d - 1 << std::endl;

        features( Eigen::seq( offset, offset + m_d - 1 ),
                  Eigen::placeholders::all )
            << X;
        offset += m_d;
    }

    std::cout << "Adding reservoir states from " << offset << " - "
              << feature_size() - 1 << std::endl;

    // Add reservoir states to feature matrix
    features( Eigen::seq( offset, Eigen::placeholders::last ),
              Eigen::placeholders::all ) = R;

    return features;
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline void
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::train( const UTIL::ConstRefMat<T> & X,
                                     const UTIL::ConstRefMat<T> & y ) noexcept {
    // Input shape checks
    if ( X.cols() != m_d || y.cols() != m_d ) {
        std::cerr << std::format(
            "ESN::train(X, y): Shape mismatch error. Input columns must match "
            "specified dimensions. (X: {}, y: {}, d: {})\n",
            X.cols(), y.cols(), m_d );
        exit( EXIT_FAILURE );
    }
    if ( X.rows() != y.rows() ) {
        std::cerr << std::format(
            "The number of observations in the provided samples (X) must match "
            "the number of observations in the labels (y). (X: {}, y: {})\n",
            X.rows(), y.rows() );
        exit( EXIT_FAILURE );
    }

    // Apply input scaling
    const auto scaled_X{ m_input_scale * X };

    // Calculate reservoir states
    const auto reservoir_states{ construct_reservoir_states(
        scaled_X.transpose() ) };

    // Construct feature vectors
    const auto feature_vectors{ construct_all_features( scaled_X.transpose(),
                                                        reservoir_states ) };

    // Store final linear input Xi
    m_xi = scaled_X( Eigen::placeholders::last, Eigen::placeholders::all )
               .transpose();

    // Calculate wi_xi
    m_wi_xi = wi_xi( m_xi );

    // Set internal reservoir state to last step
    m_reservoir =
        reservoir_states( Eigen::placeholders::all, Eigen::placeholders::last );

    if constexpr ( target_difference ) {
        const auto targets{ transform_labels( m_input_scale * y - scaled_X ) };
        m_w_out = m_solver.solve(
            feature_vectors.rightCols( feature_vectors.cols() - m_n_warmup )
                .transpose(),
            targets.bottomRows( y.rows() - m_n_warmup ) );
    }
    else {
        const auto targets{ transform_labels( y ) };
        m_w_out = m_solver.solve(
            feature_vectors.rightCols( feature_vectors.cols() - m_n_warmup )
                .transpose(),
            targets.bottomRows( y.rows() - m_n_warmup ) );
    }

    m_train_complete = true;
}

template <UTIL::Weight T, input_t W_in_init, adjacency_t A_init,
          feature_t Feature_init, UTIL::Solver S,
          UTIL::RandomNumberEngine Generator, bool target_difference,
          Eigen::StorageOptions _Options, std::signed_integral _StorageIndex>
constexpr inline UTIL::Mat<T>
ESN<T, W_in_init, A_init, Feature_init, S, Generator, target_difference,
    _Options, _StorageIndex>::forecast( const UTIL::ConstRefMat<T> &
                                            labels ) noexcept {
    // Check training has been completed first
    if ( !m_train_complete ) {
        std::cerr << std::format(
            "Call ESN::train(X, y) before forecasting, otherwise output "
            "weights are untrained.\n" );
        exit( EXIT_FAILURE );
    }

    // Shape checks
    if ( labels.cols() != m_d ) {
        std::cerr << std::format(
            "Label dimensionality ({}) must match training data ({}).\n",
            labels.cols(), m_d );
        exit( EXIT_FAILURE );
    }

    // Warmup stage
    for ( UTIL::Index i{ 0 }; i < m_n_warmup; ++i ) {
        // Exact labels used during warmup
        m_xi = m_input_scale * labels.row( i );
        m_wi_xi = wi_xi( m_xi );
        m_reservoir = R_next( m_reservoir, m_wi_xi );
    }

    // Forecasting stage
    UTIL::Mat<T> results( labels.rows() - m_n_warmup, m_d );

    for ( UTIL::Index i{ m_n_warmup }; i < labels.rows(); ++i ) {
        // Construct feature vector
        const auto feature_vector{ construct_feature( m_input_scale * m_xi,
                                                      m_reservoir ) };

        // Predict next step
        const auto prediction = m_w_out * feature_vector;

        m_xi = labels.row( i ).transpose();

        UTIL::Index j{ 0 };
        for ( const auto idx : m_train_targets ) {
            m_xi[idx] = prediction[j++];
        }

        // Write to result
        results.row( i - m_n_warmup ) = m_xi.transpose();
        m_xi *= m_input_scale;

        // Set internal wi_xi
        m_wi_xi = wi_xi( m_xi );

        // Update new reservoir state
        m_reservoir = R_next( m_reservoir, m_wi_xi );
    }

    return results;
}

} // namespace ESN
