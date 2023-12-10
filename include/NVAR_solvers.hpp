#pragma once

#include "NVAR_util.hpp"
#include "nlohmann/json.hpp"

namespace NVAR
{

template <class S, typename T>
concept Solver = requires( const S solver, const ConstRefMat<T> features,
                           const ConstRefMat<T> Y, const T alpha,
                           const nlohmann::json & params ) {
    std::floating_point<T> || is_complex<T>::value;
    { S( params ) } -> std::same_as<S>;
    { solver.solve( features, Y ) } -> std::same_as<Mat<T>>;
    { solver.reset( params ) } -> std::same_as<bool>;
};

template <Weight T>
class L1Solver
{};

template <Weight T>
class L2Solver
{
    private:
    T m_alpha;

    public:
    L2Solver( const nlohmann::json & params ) {
        try {
            m_alpha = params["alpha"].get<T>();
        }
        catch ( const nlohmann::json::parse_error & e ) {
            std::cerr << std::format(
                "Unable to construct MonomialConstructor without "
                "parameters:\n\t- alpha (Weight = std::floating_point || "
                "std::complex).\nERROR: {}\n",
                e.what() );
            exit( EXIT_FAILURE );
        }
    }

    constexpr inline Mat<T> solve( const ConstRefMat<T> features,
                                   const ConstRefMat<T> Y ) const noexcept {
        const auto feature_vec_product{ features * features.transpose() };
        const auto tikhonov_matrix{ m_alpha
                                    * Mat<T>::Identity(
                                        feature_vec_product.rows(),
                                        feature_vec_product.cols() ) };
        const auto sum{ feature_vec_product - tikhonov_matrix };
        const auto factor{
            sum.completeOrthogonalDecomposition().pseudoInverse()
        };
        return Y.transpose() * ( factor * features ).tranpose();
    }

    constexpr inline bool
    reset( const nlohmann::json & params ) const noexcept {
        bool good_params = true;

        try {
            m_alpha = params["alpha"].get<T>();
        }
        catch ( const nlohmann::json::parse_error & e ) {
            std::cerr << std::format(
                "Unable to reset MonomialConstructor without "
                "parameters:\n\t- alpha (Weight = std::floating_point || "
                "std::complex).\nERROR: {}\n",
                e.what() );
            good_params = false;
        }

        return good_params;
    }
};

} // namespace NVAR
