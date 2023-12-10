#pragma once

#include "NVAR_util.hpp"
#include "nlohmann/json.hpp"

namespace NVAR
{

template <typename C, typename T>
concept Constructor =
    requires( const C constructor, const ConstRefMat<T> source,
              const nlohmann::json & params ) {
        std::floating_point<T> || is_complex<T>::value;
        { C{ params } } -> std::same_as<C>;
        { constructor.construct_linear( source ) } -> std::same_as<Mat<T>>;
        { constructor.construct_nonlinear( source ) } -> std::same_as<Mat<T>>;
        { constructor.construct( source ) } -> std::same_as<Mat<T>>;
        { constructor.linear_size() } -> std::same_as<Index>;
        { constructor.nonlinear_size() } -> std::same_as<Index>;
        { constructor.size() } -> std::same_as<Index>;
        { constructor.reset( params ) } -> std::same_as<bool>;
    };

template <Weight T>
class MonomialConstructor
{
    private:
    // State variables
    Index m_d;
    Index m_k;
    Index m_s;
    Index m_p;
    bool  m_use_const;
    T     m_constant;

    // Indices to construct nonlinear features from linear features
    std::vector<std::vector<Index>> m_indices;

    constexpr inline void nonlinear_vec( const ConstRefVec<T> v,
                                         RefVec<T> output ) const noexcept {
        for ( Index i{ 0 }; i < v.size(); ++i ) {
            for ( const Index idx : m_indices[static_cast<std::size_t>( i )] ) {
                output[i] *= v[idx];
            }
        }
    }

    public:
    MonomialConstructor( const nlohmann::json & params ) {
        try {
            m_d = params["d"].get<Index>();
            m_k = params["k"].get<Index>();
            m_s = params["s"].get<Index>();
            m_p = params["p"].get<Index>();
            m_use_const = params["use_const"].get<bool>();
            m_constant = params["constant"].get<T>();
        }
        catch ( const nlohmann::json::parse_error & e ) {
            std::cerr << std::format(
                "Unable to construct MonomialConstructor without parameters "
                "for d (long), k (long), p (long).\nERROR: {}\n",
                e.what() );
            exit( EXIT_FAILURE );
        }
        m_indices = combinations_with_replacement( m_d * m_k, m_p );
    }

    [[nodiscard]] constexpr inline Index linear_size() const noexcept {
        return m_d * m_k;
    }
    [[nodiscard]] constexpr inline Index nonlinear_size() const noexcept {
        return factorial_20( m_d * m_k + m_p - 1 )
               / ( factorial_20( m_p ) * factorial_20( m_d * m_k - 1 ) );
    }
    [[nodiscard]] constexpr inline Index feature_size() const noexcept {
        return linear_size() + nonlinear_size() + ( m_use_const ? 1 : 0 );
    }

    [[nodiscard]] constexpr inline Mat<T>
    construct_linear( const ConstRefMat<T> source_data ) const noexcept {
        if ( m_d != source_data.cols() ) {
            std::cerr << std::format(
                "The number of columns in the source data must match the "
                "specified value of 'd' ({}).\n",
                m_d );
            exit( EXIT_FAILURE );
        }

        const Index training_inst{ source_data.rows() - m_s * ( m_k - 1 ) };
        Mat<T>      linear_features( linear_size(), training_inst );

        for ( Index i{ 0 }; i < training_inst; ++i ) {
            linear_features.col( i ) =
                construct_x_i( source_data, i, m_k, m_s );
        }

        return linear_features;
    }

    [[nodiscard]] constexpr inline Mat<T>
    construct_nonlinear( const ConstRefMat<T> linear_features ) const noexcept {
        auto nonlinear_features{ Mat<T>::Ones(
            static_cast<Index>( nonlinear_size() ), linear_features.cols() ) };

        for ( Index i{ 0 }; i < linear_features.cols(); ++i ) {
            nonlinear_vec( linear_features, nonlinear_features.col( i ) );
        }

        return nonlinear_features;
    }

    [[nodiscard]] constexpr inline bool reset( const nlohmann::json & params ) {
        bool good_params{ true };
        try {
            m_d = params["d"].get<Index>();
            m_k = params["k"].get<Index>();
            m_p = params["p"].get<Index>();
        }
        catch ( const nlohmann::json::parse_error & e ) {
            std::cerr << std::format(
                "Unable to reset MonomialConstructor without parameters:\n"
                "\t- d (long),\n\t- k (long),\n\t- p (long).\nERROR: {}\n",
                e.what() );
            good_params = false;
        }
        if ( good_params ) {
            m_indices = combinations_with_replacement( m_d * m_k, m_p );
        }

        return good_params;
    }
};

} // namespace NVAR
