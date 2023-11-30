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
#include <tuple>

#define TEST

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

template <typename T, NVAR::Index R = -1, NVAR::Index C = -1>
std::string
shape_str( const NVAR::ConstRefMat<T, R, C> m ) {
    return std::format( "({}, {})", m.rows(), m.cols() );
}

int
main( [[maybe_unused]] int argc, [[maybe_unused]] char * argv[] ) {
#ifdef TEST
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

    const bool        use_const{ true };
    const double      alpha{ 1E-6 }, constant{ 0.1 };
    const NVAR::Index d{ 3 }, k{ 3 }, s{ 2 }, p{ 3 }, delay{ 2 },
        data_stride{ 3 };
    std::cout << std::format( "data_stride: {}\n", data_stride );
    std::cout << std::format( "alpha: {}, use_const: {}, constant: {}\n", alpha,
                              use_const ? "true" : "false", constant );
    std::cout << std::format( "d = {}, k = {}, s = {}, p = {}\n", d, k, s, p );

    const NVAR::Mat<double> train_data_pool{ train_csv.atv<double>( 0, 0 ) };
    const NVAR::Mat<double> test_data_pool{ test_csv.atv<double>( 0, 0 ) };

    const std::vector<std::tuple<NVAR::Index, NVAR::Index>> feature_shape{
        { 2, 0 }, { 2, delay }, { 1, 0 }
    };

    const auto [train_samples, train_labels] = NVAR::train_split<double>(
        train_data_pool, feature_shape, k, s, data_stride );
    const auto [test_warmup, test_labels] = NVAR::test_split<double>(
        test_data_pool, feature_shape, k, s, data_stride );

    std::cout << std::format( "train_samples: {}, train_labels: {}\n",
                              shape_str<double, -1, -1>( train_samples ),
                              shape_str<double, -1, -1>( train_labels ) );
    std::cout << std::format( "test_warmup: {}, test_labels: {}\n",
                              shape_str<double, -1, -1>( test_warmup ),
                              shape_str<double, -1, -1>( test_labels ) );

    #undef FORECAST
#endif
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


    const bool        use_const{ true };
    const double      alpha{ 1E-6 }, constant{ 0.1 };
    const NVAR::Index d{ 2 }, k{ 3 }, s{ 2 }, p{ 3 }, n{ 3 }, data_stride{ 3 };
    std::cout << std::format( "data_stride: {}\n", data_stride );
    std::cout << std::format( "alpha: {}, use_const: {}, constant: {}\n", alpha,
                              use_const ? "true" : "false", constant );
    std::cout << std::format( "d = {}, k = {}, s = {}, p = {}\n", d, k, s, p );

    // Get training samples
    std::cout << "Getting training samples.\n";
    const auto              train_data_pool{ train_csv.atv<double>( 0, 0 ) };
    const NVAR::Mat<double> train_data{ train_data_pool(
        Eigen::seq( 0, Eigen::last, data_stride ), Eigen::all ) };
    std::cout << "Training keys:\n";
    int i{ 0 };
    for ( const auto & key : train_csv.col_titles() ) {
        std::cout << "\t" << i++ << ": " << key << "\n";
    }
    const NVAR::Mat<double> train_samples_0{ train_data(
        Eigen::seq( 0, Eigen::last - 1 ), 1 ) };
    std::cout << "train_samples_0, should be current i: "
              << train_samples_0( Eigen::seq( 0, n ), 0 ).transpose() << "\n";
    const NVAR::Mat<double> train_samples_1{ train_data(
        Eigen::seq( 0, Eigen::last - 1 ), 2 ) };
    std::cout << "train_samples_1, should be voltage i: "
              << train_samples_1( Eigen::seq( 0, n ), 0 ).transpose() << "\n";
    if ( train_samples_0.rows() != train_samples_1.rows() ) {
        std::cout << std::format(
            "Training samples:\n\tMismatched column lengths: {}, {}\n",
            train_samples_0.rows(), train_samples_1.rows() );
        exit( 1 );
    }
    NVAR::Mat<double> train_samples( train_samples_0.rows(), 2 );
    train_samples << train_samples_0, train_samples_1;

    // Get training labels
    std::cout << "Getting training labels.\n";
    const NVAR::Mat<double> train_labels_0{ train_data(
        Eigen::seq( 1, Eigen::last ), 1 ) };
    std::cout << "train_labels_0, should be current i: "
              << train_labels_0( Eigen::seq( 0, n ), 0 ).transpose() << "\n";
    const NVAR::Mat<double> train_labels_1{ train_data(
        Eigen::seq( 1, Eigen::last ), 2 ) };
    std::cout << "train_labels_1, should be voltage i - 1: "
              << train_labels_1( Eigen::seq( 0, n ), 0 ).transpose() << "\n";
    if ( train_labels_0.rows() != train_labels_1.rows() ) {
        std::cout << std::format(
            "Training labels:\n\tMismatched column lengths: {}, {}\n",
            train_labels_0.rows(), train_labels_1.rows() );
        exit( 1 );
    }
    NVAR::Mat<double> train_labels( train_labels_0.rows(), 2 );
    train_labels << train_labels_0, train_labels_1;

    // Get testing samples
    std::cout << "Getting testing samples.\n";
    const auto              test_data_pool{ test_csv.atv<double>( 0, 0 ) };
    const NVAR::Mat<double> test_data{ test_data_pool(
        Eigen::seq( 0, Eigen::last, data_stride ), Eigen::all ) };
    const NVAR::Mat<double> test_times{ test_data( Eigen::all, 0 ) };
    const NVAR::Mat<double> test_0{ test_data( Eigen::all, 1 ) };
    const NVAR::Mat<double> test_1{ test_data( Eigen::all, 2 ) };
    if ( test_0.rows() != test_1.rows() ) {
        std::cout << std::format( "Mismatched column lengths: {}, {}\n",
                                  test_0.rows(), test_1.rows() );
        exit( 1 );
    }
    NVAR::Mat<double> test_sample_pool( test_0.rows(), 2 );
    test_sample_pool << test_0, test_1;

    const NVAR::Mat<double> test_warmup{ test_sample_pool(
        Eigen::seqN( 0, s * ( k - 1 ) + 1 ), Eigen::all ) };
    const NVAR::Mat<double> test_labels{ test_sample_pool(
        Eigen::seq( s * ( k - 1 ) + 2, Eigen::last ), Eigen::all ) };
    const NVAR::Mat<double> test_label_times{ test_times(
        Eigen::seq( s * ( k - 1 ) + 2, Eigen::last ), 0 ) };

    std::cout << "Training NVAR.\n";
    NVAR::NVAR_runtime<double, NVAR::nonlinear_t::poly> test(
        train_samples, train_labels, d, k, s, p, 0.001, true, double{ 0.1 } );


    std::cout << "Forecasting.\n";
    auto forecast{ test.forecast( test_warmup, test_labels,
                                  std::vector<NVAR::Index>{ 0 } ) };

    // Write results out
    std::cout << "Writing results.\n";
    const std::filesystem::path forecast_path{
        "../data/forecast_data/forecast.csv"
    };
    const std::vector<std::string> forecast_col_titles{ "t", "I", "V", "I'",
                                                        "V'" };
    std::cout << std::format(
        "forecast_col_titles.size(): {}, times.cols(): {}, forecast.cols(): "
        "{}, test_labels.cols(): {}\n",
        forecast_col_titles.size(), test_label_times.cols(), forecast.cols(),
        test_labels.cols() );
    NVAR::Mat<double> forecast_data( forecast.rows(),
                                     forecast.cols() + test_labels.cols()
                                         + test_label_times.cols() );

    forecast_data << test_label_times, forecast, test_labels;
    NVAR::SimpleCSV::write<double>( forecast_path, forecast_data,
                                    forecast_col_titles );
#endif
#ifdef CUSTOM_FEATURES
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

    const bool        use_const{ true };
    const double      alpha{ 1 }, constant{ 1 };
    const NVAR::Index d{ 3 }, k{ 3 }, s{ 2 }, p{ 1 }, n{ 3 }, data_stride{ 3 },
        delay{ 1 };
    std::cout << std::format( "data_stride: {}, delay: {}\n", data_stride,
                              delay );
    std::cout << std::format( "alpha: {}, use_const: {}, constant: {}\n", alpha,
                              use_const ? "true" : "false", constant );
    std::cout << std::format( "d = {}, k = {}, s = {}, p = {}\n", d, k, s, p );

    // Get training samples
    std::cout << "Getting training samples.\n";
    const auto              train_data_pool{ train_csv.atv<double>( 0, 0 ) };
    const NVAR::Mat<double> train_data{ train_data_pool(
        Eigen::seq( 0, Eigen::last, data_stride ), Eigen::all ) };
    std::cout << "Training keys:\n";
    int i{ 0 };
    for ( const auto & key : train_csv.col_titles() ) {
        std::cout << "\t" << i++ << ": " << key << "\n";
    }
    const NVAR::Mat<double> train_samples_0{ train_data(
        Eigen::seq( delay, Eigen::last - 1 ), 2 ) };
    std::cout << "train_samples_0, should be voltage i: "
              << train_samples_0( Eigen::seq( 0, n ), 0 ).transpose() << "\n";
    const NVAR::Mat<double> train_samples_1{ train_data(
        Eigen::seq( 0, Eigen::last - delay - 1 ), 2 ) };
    std::cout << "train_samples_1, should be voltage i - 1: "
              << train_samples_1( Eigen::seq( 0, n ), 0 ).transpose() << "\n";
    const NVAR::Mat<double> train_samples_2{ train_data(
        Eigen::seq( delay, Eigen::last - 1 ), 1 ) };
    std::cout << "train_samples_2, should be current i: "
              << train_samples_2( Eigen::seq( 0, n ), 0 ).transpose() << "\n";
    if ( train_samples_0.rows() != train_samples_1.rows()
         || train_samples_1.rows() != train_samples_2.rows() ) {
        std::cout << std::format(
            "Training samples:\n\tMismatched column lengths: {}, {}, {}\n",
            train_samples_0.rows(), train_samples_1.rows(),
            train_samples_2.rows() );
        exit( 1 );
    }
    NVAR::Mat<double> train_samples( train_samples_0.rows(), 3 );
    train_samples << train_samples_0, train_samples_1, train_samples_2;

    // Get training labels
    std::cout << "Getting training labels.\n";
    const NVAR::Mat<double> train_labels_0{ train_data(
        Eigen::seq( delay + 1, Eigen::last ), 2 ) };
    std::cout << "train_labels_0, should be voltage i: "
              << train_labels_0( Eigen::seq( 0, n ), 0 ).transpose() << "\n";
    const NVAR::Mat<double> train_labels_1{ train_data(
        Eigen::seq( 1, Eigen::last - delay ), 2 ) };
    std::cout << "train_labels_1, should be voltage i - 1: "
              << train_labels_1( Eigen::seq( 0, n ), 0 ).transpose() << "\n";
    const NVAR::Mat<double> train_labels_2{ train_data(
        Eigen::seq( delay + 1, Eigen::last ), 1 ) };
    std::cout << "train_labels_2, should be current i: "
              << train_labels_2( Eigen::seq( 0, n ), 0 ).transpose() << "\n";
    if ( train_labels_0.rows() != train_labels_1.rows()
         || train_labels_1.rows() != train_labels_2.rows() ) {
        std::cout << std::format(
            "Training labels:\n\tMismatched column lengths: {}, {}, {}\n",
            train_labels_0.rows(), train_labels_1.rows(),
            train_labels_2.rows() );
        exit( 1 );
    }
    NVAR::Mat<double> train_labels( train_labels_0.rows(), 3 );
    train_labels << train_labels_0, train_labels_1, train_labels_2;

    // Get testing samples
    std::cout << "Getting testing samples.\n";
    const auto              test_data_pool{ test_csv.atv<double>( 0, 0 ) };
    const NVAR::Mat<double> test_data{ test_data_pool(
        Eigen::seq( 0, Eigen::last, data_stride ), Eigen::all ) };
    const NVAR::Mat<double> test_times{ test_data(
        Eigen::seq( delay, Eigen::last ), 0 ) };
    const NVAR::Mat<double> test_0{ test_data( Eigen::seq( delay, Eigen::last ),
                                               2 ) };
    const NVAR::Mat<double> test_1{ test_data(
        Eigen::seq( 0, Eigen::last - delay ), 2 ) };
    const NVAR::Mat<double> test_2{ test_data( Eigen::seq( delay, Eigen::last ),
                                               1 ) };
    if ( test_0.rows() != test_1.rows() || test_1.rows() != test_2.rows() ) {
        std::cout << std::format( "Mismatched column lengths: {}, {}, {}\n",
                                  test_0.rows(), test_1.rows(), test_2.rows() );
        exit( 1 );
    }
    NVAR::Mat<double> test_sample_pool( test_0.rows(), 3 );
    test_sample_pool << test_0, test_1, test_2;

    const NVAR::Mat<double> test_warmup{ test_sample_pool(
        Eigen::seqN( 0, s * ( k - 1 ) ), Eigen::all ) };
    const NVAR::Mat<double> test_labels{ test_sample_pool(
        Eigen::seq( s * ( k - 1 ) + 1, Eigen::last ), Eigen::all ) };
    const NVAR::Mat<double> test_label_times{ test_times(
        Eigen::seq( s * ( k - 1 ) + 1, Eigen::last ), 0 ) };

    // Building NVAR
    std::cout << "Building NVAR.\n";
    NVAR::NVAR_runtime<double, NVAR::nonlinear_t::poly> test(
        train_samples, train_labels, d, k, s, p, alpha, use_const,
        double{ constant } );

    std::cout << "Forecasting.\n";
    auto forecast{ test.forecast( test_warmup, test_labels,
                                  std::vector<NVAR::Index>{ 2 } ) };

    // Write results out
    std::cout << "Writing results.\n";
    const std::filesystem::path forecast_path{
        "../data/forecast_data/forecast.csv"
    };
    const std::vector<std::string> forecast_col_titles{
        "t", "V_(n)", "V_(n-1)", "I_(n)", "V_(n)'", "V_(n-1)'", "I_(n)'"
    };
    std::cout << std::format(
        "forecast_col_titles.size(): {}, times.cols(): {}, forecast.cols(): "
        "{}, test_labels.cols(): {}\n",
        forecast_col_titles.size(), test_label_times.cols(), forecast.cols(),
        test_labels.cols() );
    NVAR::Mat<double> forecast_data( forecast.rows(),
                                     forecast.cols() + test_labels.cols()
                                         + test_label_times.cols() );

    forecast_data << test_label_times, forecast, test_labels;
    NVAR::SimpleCSV::write<double>( forecast_path, forecast_data,
                                    forecast_col_titles );

#endif
#ifdef DOUBLESCROLL
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

    const NVAR::Index d2{ 3 }, k2{ 2 }, s2{ 1 }, p2{ 3 };
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
        0.01, false, double{ 1. } );

    // Forecast
    auto doublescroll_forecast{ doublescroll_test.forecast(
        doublescroll_warmup, doublescroll_test_labels, { 0 } ) };

    // Write data
    const std::filesystem::path doublescroll_forecast_path{
        "../data/forecast_data/"
        "doublescroll_predict.csv"
    };
    const std::vector<std::string> col_titles{ "v1", "v2", "I" };
    NVAR::SimpleCSV::write<double>( doublescroll_forecast_path,
                                    doublescroll_forecast, col_titles );
#endif
#ifdef HH_MODEL
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
