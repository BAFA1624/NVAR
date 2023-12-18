#pragma once

#include "ESN_util.hpp"
#include "nlohmann/json.hpp"

namespace ESN
{

template <Weight T, input_init_t W_in_init = input_init_t::init_default,
          RandomNumberEngine Generator = std::mersenne_twister_engine<
              unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,
              0x9d2c5680, 15, 0xefc60000, 18, 1812433253>>
class ESN
{
    private:
    typedef typename Generator::result_type Seed_t;

    // State variables for the ESN
    Index  m_d;
    Index  m_n_node;
    T      m_leak;
    T      m_sparsity;
    T      m_spectral_radius;
    Seed_t m_seed;

    constexpr static bool m_dense_input =
        static_cast<bool>( W_in_init & input_init_t::dense );

    // Matrices for the ESN
    Vec<T>  m_reservoir;
    SMat<T> m_w_in;
    Mat<T>  m_w_in_dense;
    Mat<T>  m_w_out;
    Mat<T>  m_adjacency;

    // Common random number generator for all random processes
    Generator m_gen;

    // Initialisation functions to generate W_in & reservoir node values
    std::function<T( const T, Generator & )> & m_W_in_func;
    std::function<T( const T, Generator & )> & m_res_func;
    std::function<T( const T )> &              m_activation;

    [[nodiscard]] constexpr inline SMat<T> init_W_in() const noexcept;
    [[nodiscard]] constexpr inline Mat<T>  init_W_in_dense() const noexcept;
    [[nodiscard]] constexpr inline Mat<T>  init_adjacency() const noexcept;
    [[nodiscard]] constexpr inline Vec<T>  init_res() const noexcept;
    [[nodiscard]] constexpr inline Vec<T>
    R_next( const ConstRefVec<T> R ) const noexcept;
    [[nodiscard]] constexpr inline Mat<T>
    train_W_out( const ConstRefMat<T> outputs,
                 const ConstRefMat<T> labels ) const noexcept;

    public:
    ESN() = delete;
    ESN( const nlohmann::json & params );
    ESN(
        const Index d, const Index n_node, const T leak, const T sparsity,
        const T                               spectral_radius,
        const typename Generator::result_type seed =
            typename Generator::result_type{ 0 },
        const std::function<T( const T )> & activation =
            []( const T x ) { return std::tanh( x ); },
        const std::function<T( const T, Generator & )> & W_in_func =
            []( [[maybe_unused]] const T x, [[maybe_unused]] Generator & gen ) {
                static auto dist{ std::uniform_real_distribution<T>( -1.,
                                                                     1. ) };
                return dist( gen );
            },
        const std::function<T( const T, Generator & )> & res_func =
            []( [[maybe_unused]] const T x, [[maybe_unused]] Generator & gen ) {
                return static_cast<T>( 0. );
            } );

    // Getters
    [[nodiscard]] constexpr inline ConstRefVec<T> R() const noexcept {
        return m_reservoir;
    }
    [[nodiscard]] constexpr inline ConstRefSMat<T> W_in() const noexcept {
        return m_w_in;
    }
    [[nodiscard]] constexpr inline ConstRefMat<T> W_in_dense() const noexcept {
        return m_w_in_dense;
    }
    [[nodiscard]] constexpr inline ConstRefMat<T> W_out() const noexcept {
        return m_w_out;
    }
    [[nodiscard]] constexpr inline ConstRefMat<T> A() const noexcept {
        return m_adjacency;
    }

    [[nodiscard]] constexpr inline Mat<T>
    forecast( const ConstRefVec<T> warmup, const ConstRefVec<T> labels,
              const std::vector<Index> & pass_through = {} ) const noexcept;
};

template <Weight T, input_init_t W_in_init, RandomNumberEngine Generator>
ESN<T, W_in_init, Generator>::ESN(
    const Index d, const Index n_node, const T leak, const T sparsity,
    const T spectral_radius, const typename Generator::result_type seed,
    const std::function<T( const T )> &              activation,
    const std::function<T( const T, Generator & )> & W_in_func,
    const std::function<T( const T, Generator & )> & res_func ) :
    m_d( d ),
    m_n_node( n_node ),
    m_leak( leak ),
    m_sparsity( sparsity ),
    m_spectral_radius( spectral_radius ),
    m_seed( seed ),
    m_gen( seed ),
    m_activation( activation ),
    m_W_in_func( W_in_func ),
    m_res_func( res_func ) {
    m_gen.discard( static_cast<unsigned long long>( m_gen() ) );

    // Initialise input weights
    // Check that one of input_init_t::split or input_init_t::homogeneous is
    // defined, but not both.
    static_assert(
        static_cast<bool>( ( W_in_init & input_init_t::split )
                           ^ ( W_in_init & input_init_t::homogeneous ) ),
        "Define either input_init_t::split OR input_init_t::homogeneous.\n" );

    // Check that one of input_init_t::sparse or input_init_t::dense is define,
    // but not both.
    static_assert(
        static_cast<bool>( ( W_in_init & input_init_t::sparse )
                           ^ ( W_in_init & input_init_t::dense ) ),
        "Define either input_init_t::sparse OR input_init_t::dense.\n" );

    if constexpr ( m_dense_input ) {
        m_w_in = init_W_in();
    }
    else {
        m_w_in_dense = init_W_in_dense();
    }

    m_adjacency = init_adjacency();
}


template <Weight T, input_init_t W_in_init, RandomNumberEngine Generator>
constexpr inline SMat<T>
ESN<T, W_in_init, Generator>::init_W_in() const noexcept {
    constexpr bool split{ static_cast<bool>( W_in_init
                                             & input_init_t::split ) };

    return generate_sparse<T, split, Generator>( m_n_node, m_d, m_sparsity,
                                                 m_gen, m_W_in_func );
}

template <Weight T, input_init_t W_in_init, RandomNumberEngine Generator>
constexpr inline Mat<T>
ESN<T, W_in_init, Generator>::init_W_in_dense() const noexcept {
    constexpr bool split{ static_cast<bool>( W_in_init
                                             & input_init_t::split ) };

    Mat<T> result( m_n_node, m_d );

    const auto f = [this]( const T x ) { return m_res_func( x, m_gen ); };
    if constexpr ( split ) {
        std::vector<Index> sizes( static_cast<std::size_t>( m_d ),
                                  m_n_node / m_d );
        for ( std::size_t i{ 0 };
              i < static_cast<std::size_t>( m_n_node % m_d ); ++i ) {
            sizes[i]++;
        }

        Index offset{ 0 };
        for ( const auto [i, n_rows] : sizes | std::views::enumerate ) {
            result( Eigen::seq( offset, offset + n_rows - 1 ),
                    static_cast<Index>( i ) )
                .unaryExpr( f );

            offset += n_rows;
        }
    }
    else {
        result.unaryExpr( f );
    }

    return result;
}

template <Weight T, input_init_t W_in_init, RandomNumberEngine Generator>
constexpr inline Mat<T>
ESN<T, W_in_init, Generator>::init_adjacency() const noexcept {
    auto A = generate_sparse<T, false, Generator>( m_n_node, m_n_node,
                                                   m_sparsity, m_gen );

    return A;
}

} // namespace ESN
