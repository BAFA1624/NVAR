#include "Eigen/Dense"
#include "NVAR.hpp"
#include "nlohmann/json.hpp"
#include "simple_csv.hpp"

#include <array>  //exec
#include <cstdio> // exec
#include <format>
#include <iostream>
#include <memory>    // exec
#include <stdexcept> // exec
#include <string>    // exec

template <typename T>
void
print( const std::vector<T> & v ) {
    for ( const auto & x : v ) { std::cout << x << "  "; }
    std::cout << std::endl;
}

std::tuple<int, std::string>
exec( const char * const cmd ) {
    std::string full_cmd{ cmd };
    full_cmd += " 2>&1";

    std::array<char, 128>                      buffer;
    std::unique_ptr<FILE, decltype( &pclose )> pipe(
        popen( full_cmd.c_str(), "r" ), pclose );

    if ( !pipe ) {
        return { 1, std::string{ "popen() failed to open." } };
    }

    std::string std_output;
    while ( fgets( buffer.data(), buffer.size(), pipe.get() ) != nullptr ) {
        std_output += buffer.data();
    }

    return { 0, std_output };
}

int
main( [[maybe_unused]] int argc, [[maybe_unused]] char * argv[] ) {
#ifdef FORECAST
    std::cout << "Running NVAR." << std::endl;

    const std::filesystem::path train_path{
        //"../data/train_data/17_0_-1_10000_0_1_1.csv"
        "../data/train_data/21_0_-1_21000_0_1_1.csv"
    };
    const std::filesystem::path test_path{
        "../data/test_data/21_1_-1_189000_0_1_1.csv"
    };

    const auto train_details = NVAR::parse_filename( train_path.string() );
    const auto test_details = NVAR::parse_filename( test_path.string() );

    const auto train_csv{ NVAR::SimpleCSV(
        /*filename=*/train_path,
        /*col_titles=*/true,
        /*skip_header=*/1,
        /*delim*/ ",",
        /*max_line_size=*/256 ) };
    const auto test_csv{ NVAR::SimpleCSV(
        /*filename=*/test_path,
        /*col_titles=*/true,
        /*skip_header=*/1,
        /*delim=*/",",
        /*max_line_size=*/256 ) };


    const NVAR::Index                  d{ 2 }, k{ 3 }, s{ 3 }, p{ 2 };
    [[maybe_unused]] const NVAR::Index n{ -1 };
    std::cout << std::format( "d = {}, k = {}, s = {}, p = {}, n = {}\n", d, k,
                              s, p, n );

    const auto              train_data{ train_csv.atv<double>( 0, 0 ) };
    const NVAR::Mat<double> train_samples{ train_data( Eigen::all,
                                                       Eigen::seq( 1, 2 ) ) };
    const NVAR::Mat<double> train_labels{ train_data(
        Eigen::seq( ( k - 1 ) * s, Eigen::last ), Eigen::seq( 3, 4 ) ) };


    const auto              test_data{ test_csv.atv<double>( 0, 0 ) };
    const NVAR::Mat<double> test_warmup{ test_data(
        Eigen::seqN( ( k - 1 ) * s, k * s - ( s - 1 ), -1 ),
        Eigen::seq( 1, 2 ) ) };
    const NVAR::Mat<double> test_samples{ test_data(
        Eigen::seq( s * ( k - 1 ) + 2, Eigen::last - 1 ),
        Eigen::seq( 1, 2 ) ) };
    const NVAR::Mat<double> test_labels{ test_data(
        Eigen::seq( s * ( k - 1 ) + 3, Eigen::last ), Eigen::seq( 3, 4 ) ) };

    std::cout << "Training NVAR.\n";
    NVAR::NVAR_runtime<double, NVAR::nonlinear_t::poly> test(
        train_samples, train_labels, d, k, s, p, 1E-5, true, double{ 0.1 } );


    std::cout << "Forecasting.\n";
    auto forecast{ test.forecast( test_warmup, test_labels,
                                  std::vector<NVAR::Index>{ 0 } ) };

    const std::filesystem::path forecast_path{
        "../data/forecast_data/forecast.csv"
    };
    const std::vector<std::string> forecast_col_titles{ "t", "I", "V" };
    NVAR::Mat<double> forecast_data( forecast.rows(), forecast.cols() + 1 );
    std::cout << std::format( "forecast: ({}, {})\n", forecast.rows(),
                              forecast.cols() );
    std::cout << std::format(
        "times: ({}, {})\n",
        test_data( Eigen::seq( s * ( k - 1 ) + 3, Eigen::last ), 0 ).rows(),
        test_data( Eigen::seq( s * ( k - 1 ) + 3, Eigen::last ), 0 ).cols() );
    forecast_data << test_data( Eigen::seq( s * ( k - 1 ) + 3, Eigen::last ),
                                0 ),
        forecast;
    NVAR::SimpleCSV::write<double>( forecast_path, forecast_data,
                                    forecast_col_titles );

#elifdef DOUBLESCROLL
    std::cout << "Running doublescroll." << std::endl;

    // Doublescroll
    const std::filesystem::path doublescroll_train_path{
        "../data/train_data/doublescroll.csv"
    };
    const std::filesystem::path doublescroll_test_path{
        "../data/test_data/doublescroll.csv"
    };
    // clang-format off
    const auto doublescroll_train_csv{ NVAR::SimpleCSV(
        /*filename=*/doublescroll_train_path,
        /*col_titles=*/true,
        /*skip_header=*/0,
        /*delim*/ ",",
        /*max_line_size=*/256 ) };
    const auto doublescroll_test_csv{ NVAR::SimpleCSV(
        /*filename=*/doublescroll_test_path,
        /*col_titles=*/true,
        /*skip_header=*/0,
        /*delim*/ ",",
        /*max_line_size=*/256 ) };
    // clang-format on

    const auto doublescroll_train_data{ doublescroll_train_csv.atv<double>() };
    const auto doublescroll_test_data{ doublescroll_test_csv.atv<double>() };

    const NVAR::Index d2{ 3 }, k2{ 4 }, s2{ 4 }, p2{ 5 };
    const NVAR::Index n_warmup{ s2 * ( k2 - 1 ) };

    // Split training samples & labels
    const auto doublescroll_train_samples{ doublescroll_train_data(
        Eigen::seq( 0, Eigen::last - 1 ), Eigen::seq( 1, 3 ) ) };
    const auto doublescroll_train_labels{ doublescroll_train_data(
        Eigen::seq( ( k2 - 1 ) * s2 + 1, Eigen::last ), Eigen::seq( 1, 3 ) ) };

    // Pick out warmup from testing set
    const auto doublescroll_warmup{ doublescroll_test_data(
        Eigen::seq( 0, n_warmup ), Eigen::seq( 1, 3 ) ) };

    // Split testing samples & labels
    const auto doublescroll_test_samples{ doublescroll_test_data(
        Eigen::seq( n_warmup + 1, Eigen::last - 1 ), Eigen::seq( 1, 3 ) ) };
    const auto doublescroll_test_labels{ doublescroll_test_data(
        Eigen::seq( n_warmup + 2, Eigen::last ), Eigen::seq( 1, 3 ) ) };

    // Create NVAR model
    NVAR::NVAR_runtime<double> doublescroll_test(
        doublescroll_train_samples, doublescroll_train_labels, d2, k2, s2, p2,
        0.0001, false, double{ 1. } );

    // Forecast
    auto doublescroll_forecast{ doublescroll_test.forecast(
        doublescroll_warmup, doublescroll_test_labels, { 0, 1 } ) };

    // Write data
    const std::filesystem::path doublescroll_forecast_path{
        "../data/forecast_data/"
        "doublescroll_predict.csv"
    };
    const std::vector<std::string> col_titles{ "v1", "v2", "I" };
    NVAR::SimpleCSV::write<double>( doublescroll_forecast_path,
                                    doublescroll_forecast, col_titles );
#elifdef HH_MODEL
    const auto base_path{ std::filesystem::absolute(
        std::filesystem::current_path() / ".." ) };

    const std::vector<std::string> files{ "a1t01", "a3t05", "a7t10", "a8t18" };

    for ( const auto & file : files ) {
        const auto python_file = base_path / "data" / "decompress_signal.py";
        const auto read_path = base_path / "data" / "hh_data" / file;
        const auto write_path = base_path / "tmp" / ( file + ".csv" );

        std::cout << "Reading from: " << read_path << std::endl;
        std::cout << "Writing to: " << write_path << std::endl;

        const std::string decompress_cmd{ std::format(
            "python3.11 {} {} {}", python_file.string(), read_path.string(),
            write_path.string() ) };
        const auto [decompress_result_code, decompress_msg] =
            exec( decompress_cmd.c_str() );

        std::cout << std::format( "{}:\nResult code: {}\nMessage:\n{}\n", file,
                                  decompress_result_code, decompress_msg );

        if ( decompress_result_code == 0 /* Success code */ ) {
            // Create training data
            // Create warmup
            // Create test data
            // Train NVAR
            // Forecast forwards
            // Write result file
            // Delete decompressed file
            // const std::string remove_cmd{ std::format( "rm {}",
            //                                           write_path.string() )
            //                                           };
            // const auto [result_code, rm_std_output] =
            //    exec( remove_cmd.c_str() );
            // if ( result_code != 0 ) {
            //    std::cout << "Failed to remove decompressed file." <<
            //    std::endl; std::cout << "Message:\n" << rm_std_output << "\n";
            //    break;
            //}
            // std::cout << "Message:\n" << rm_std_output << "\n";
        }
    }
#endif
    return 0;
}
