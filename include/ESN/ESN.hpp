#pragma once

#include "ESN_util.hpp"

namespace ESN
{

template <Weight T>
class ESN
{
    private:
    Vec<T> m_reservoir;
    Mat<T> m_w_in;
    Mat<T> m_w_out;
    Mat<T> m_adjacency;

    public:
    ESN() {}

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
};

} // namespace ESN
