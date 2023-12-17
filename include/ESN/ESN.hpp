#pragma once

#include "ESN_util.hpp"
#include "nlohmann/json.hpp"

namespace ESN
{

template <Weight T, input_init_t W_in_init,
          RandomNumberEngine Generator = std::mersenne_twister_engine<
              unsigned, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7,
              0x9d2c5680, 15, 0xefc60000, 18, 1812433253>>
class ESN
{
    private:
    typedef typename Generator::result_type Seed_t;

    Index  m_n_node;
    T      m_leak;
    T      m_spectral_radius;
    Seed_t m_seed;

    Vec<T> m_reservoir;
    Mat<T> m_w_in;
    Mat<T> m_w_out;
    Mat<T> m_adjacency;

    std::function<T( const T, Generator & )> & m_init_func;

    [[nodiscard]] constexpr inline SMat<T> init_W_in() const noexcept;
    [[nodiscard]] constexpr inline SMat<T> init_adjacency() const noexcept;
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
        const Index n_node, const T leak, const T spectral_radius,
        const typename Generator::result_type seed =
            typename Generator::result_type{ 0 },
        const std::function<T( const T, Generator & )> & gen_value =
            []( [[maybe_unused]] const T x, [[maybe_unused]] Generator & y ) {
                static auto dist{ std::uniform_real_distribution<T>( 0., 1. ) };
                return dist( y );
            } );

    // Getters
    [[nodiscard]] constexpr inline ConstRefVec<T> R() const noexcept {
        return m_reservoir;
    }
    [[nodiscard]] constexpr inline ConstRefMat<T> W_in() const noexcept {
        return m_w_in;
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
constexpr inline SMat<T>
ESN<T, W_in_init, Generator>::init_W_in() const noexcept {
    if constexpr ( W_in_init == input_init_t::homogeneous ) {
        return generate_sparse<T, Generator>();
    }
    else if constexpr ( W_in_init == input_init_t::split ) {}
    else {
        std::cerr << std::format(
            "WARNING: Specified input weight initialization scheme ({}) is "
            "unimplemented.\n",
            static_cast<Index>( W_in_init ) );
        exit( EXIT_FAILURE );
    }
}

} // namespace ESN
