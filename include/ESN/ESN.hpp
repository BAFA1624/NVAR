#pragma once

#include "ESN_util.hpp"

namespace ESN
{

template <Weight T>
class ESN
{
    private:
    Index m_n_node;
    T     m_leak;
    T     m_spectral_radius;

    Vec<T> m_reservoir;
    Mat<T> m_w_in;
    Mat<T> m_w_out;
    Mat<T> m_adjacency;

    [[nodiscard]] constexpr inline Vec<T>
    R_next( const ConstRefVec<T> R ) const noexcept;
    [[nodiscard]] constexpr inline Mat<T>
    train_W_out( const ConstRefMat<T> outputs,
                 const ConstRefMat<T> labels ) const noexcept;

    public:
    ESN( const ConstRefVec<T> features, const ConstRefVec<T> labels,
         const T leak ) {}

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

} // namespace ESN
