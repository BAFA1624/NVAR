#include "NVAR.hpp"

#include <cassert>
#include <iostream>
#include <string_view>

namespace NVAR
{
// clang-format off
const Mat<double, 5, 4> nvar_sample_1{
    { 1,  2,  3,  4  },
    { 5,  6,  7,  8  },
    { 9,  10, 11, 12 },
    { 13, 14, 15, 16 },
    { 17, 18, 19, 20 }
};
const Mat<double, 3, 4> nvar_label_1{
    { 17, 18, 19, 20 },
    { 9,  10, 11, 12 },
    { 1,  2,  3,  4  }
};

const Mat<double> nvar_sample_2{
    { 1,  2,  3,  4  },
    { 5,  6,  7,  8  },
    { 9,  10, 11, 12 },
    { 13, 14, 15, 16 },
    { 17, 18, 19, 20 }
};
const Vec<double> nvar_label_2{
    { 17 },
    { 9 },
    { 1 },
    { 18 },
    { 10 },
    { 2 },
    { 19 },
    { 11 },
    { 3 },
    { 20 },
    { 12 },
    { 4 }
};                    
const Mat<double> nvar_sample_3{
    {1,2},
    {3,4},
    {5,6},
    {7,8},
    {9,10}
};
const Vec<double> nvar_label_3{
    {9},
    {5},
    {1},
    {10},
    {6},
    {2}
};

const Mat<double> nvar_sample_4_a{
    {1, 2},
    {3, 4},
    {5, 6},
    {7, 8},
    {9, 10}
};
const Mat<double> nvar_sample_4_b{
    {11, 12}
};
const Mat<double> nvar_label_4{
    {3,  4},
    {5,  6},
    {7,  8},
    {9,  10},
    {11, 12}
};

const Vec<double, 3> nvar_sample_5_a{
    1, 2, 3
};
const Vec<double, -1> nvar_sample_5_b{};
const Vec<double, 3> nvar_label_5{
    1, 2, 3
};

// clang-format on

} // namespace NVAR


template <typename T>
concept Printable = requires( T t ) {
    { std::cout << t } -> std::same_as<std::ostream &>;
};

template <Printable T>
constexpr void
output_err( const std::string_view msg, const T & expected, const T & result ) {
    std::cerr << msg << std::endl;
    std::cerr << "==== EXPECTED ====" << std::endl;
    std::cerr << expected << std::endl;
    std::cerr << "================" << std::endl;
    std::cerr << "==== RESULT ====" << std::endl;
    std::cerr << result << std::endl;
    std::cerr << "================" << std::endl;
}

template <Printable OutputType>
constexpr auto
test( const OutputType & result, const OutputType & expected,
      const std::string_view msg = "" ) {
    if ( result != expected ) {
        output_err( msg, result, expected );
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int
main() {
    // Test NVAR_util
    // Compile time testing
    static_assert( NVAR::factorial_20( 5 ) == 120 );
    static_assert(
        NVAR::def_total_size<3, 2, 2, NVAR::nonlinear_t::poly, true>() == 28 );
    static_assert(
        NVAR::def_total_size<3, 2, 3, NVAR::nonlinear_t::poly, false>() == 62 );
    static_assert( NVAR::def_nonlinear_size<1, 2, 3, NVAR::nonlinear_t::poly>()
                   == 4 );
    // Runtime testing
    assert( NVAR::factorial_20( 5 ) == 120 );
    assert( NVAR::def_total_size( 3, 2, 2, NVAR::nonlinear_t::poly, true )
            == 28 );
    assert( NVAR::def_total_size( 3, 2, 3, NVAR::nonlinear_t::poly, false )
            == 62 );
    assert( NVAR::def_nonlinear_size( 1, 2, 3, NVAR::nonlinear_t::poly ) == 4 );
    // TODO: combinations_with_replacement_rec 1 & 2

    // Testing 3 construct_x_i implementations
    const auto test1{ test(
        NVAR::construct_x_i<double, 4, 3, 2, 5>( NVAR::nvar_sample_1, 0 ),
        NVAR::nvar_label_1, "construct_x_i<T, d, k, s, N>() failed." ) };
    if ( test1 == EXIT_FAILURE ) {
        return test1;
    }

    const auto test2{ test(
        NVAR::construct_x_i<double>( NVAR::nvar_sample_2, 0, 3, 2 ),
        NVAR::nvar_label_2, "construct_x_i<T>(input, i, k, s) failed." ) };
    if ( test2 == EXIT_FAILURE ) {
        return test2;
    }

    const auto test3{ test(
        NVAR::construct_x_i<double>( NVAR::nvar_sample_3, 3, 2 ),
        NVAR::nvar_label_3, "construct_x_i<T>(input, i, k, s) failed." ) };
    if ( test3 == EXIT_FAILURE ) {
        return test3;
    }

    // Testing NVAR::Pipe
    // Fixed size Pipes
    // NVAR::Pipe<double, /*Transparent=*/true, /*d=*/3, /*n=*/-1>
    //           transparent_fixed_pipe{ NVAR::nvar_sample_5_b };
    // const auto test5{ test( transparent_fixed_pipe.forward(
    //                            NVAR::nvar_sample_5_a, NVAR::nvar_sample_5_b
    //                            ),
    //                        NVAR::nvar_label_5 ) };

    // Test NVAR

    return 0;
}
