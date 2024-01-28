#include "CSV/simple_csv.hpp"
#include "ESN/ESN.hpp"
#include "NVAR/NVAR.hpp"
#include "nlohmann/json.hpp"

#include <array> //exec
#include <chrono>
#include <cstdio> // exec
#include <format>
#include <iostream>
#include <memory> // exec
// #include <stdexcept> // exec
#include <string> // exec
#include <tuple>

template <typename T>
concept Printable = requires( const T x ) { std::cout << x; };

template <Printable T>
std::ostream &
operator<<( std::ostream & stream, const std::vector<T> & v ) {
    stream << "[ ";
    for ( const auto & x : v | std::views::take( v.size() - 1 ) ) {
        stream << x << " ";
    }
    stream << v.back() << " ]";
    return stream;
}

template <Printable T>
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

template <UTIL::Weight T, UTIL::Index R = -1, UTIL::Index C = -1>
std::string
shape_str( const UTIL::ConstRefMat<T, R, C> m ) {
    return std::format( "({}, {})", m.rows(), m.cols() );
}

template <UTIL::Weight T, UTIL::RandomNumberEngine Generator>
T
normal_0_2( [[maybe_unused]] const T x, Generator & gen ) {
    static std::normal_distribution<T> dist( 20., 2.0 );
    return dist( gen );
}

template <UTIL::Weight Scalar, typename Matrix>
inline static std::vector<std::vector<Scalar>>
from_eigen_matrix( const Matrix & M ) {
    std::vector<std::vector<Scalar>> m;
    m.resize( M.rows(), std::vector<Scalar>( M.cols(), 0 ) );
    for ( size_t i = 0; i < m.size(); i++ ) {
        for ( size_t j = 0; j < m.front().size(); j++ ) { m[i][j] = M( i, j ); }
    }
    return m;
}

inline auto
get_filename( const std::filesystem::path & trial_name ) {
    if ( std::filesystem::directory_entry( trial_name ).exists() ) {
        const auto now{ std::chrono::system_clock::now() };
        const auto ymd{ std::chrono::floor<std::chrono::days>( now ) };
        const auto hms{ std::chrono::hh_mm_ss( now - ymd ) };

        const auto result{ std::filesystem::path{
            trial_name.parent_path()
            / std::format( "{}_{}_{}{}", trial_name.stem().string(), ymd, hms,
                           trial_name.extension().string() ) } };

        std::cout << result << std::endl;

        return result;
    }
    else {
        return trial_name;
    }
}

int
main( [[maybe_unused]] int argc, [[maybe_unused]] char * argv[] ) {
    const auto n_thread{ Eigen::nbThreads() };
    std::cout << std::format( "Running with {} threads.\n", n_thread );
    Eigen::setNbThreads( 8 );
// #define ESN_OPT
#ifdef ESN_OPT
    using T = float;
    using clock = std::chrono::steady_clock;

    // Constant hyperparams
    constexpr UTIL::Index d{ 2 }, data_stride{ 3 };
    constexpr T           bias{ 1. };
    const std::vector<std::tuple<UTIL::Index, UTIL::Index>> feature_shape{
        { 0, 0 }, { 1, 0 }, { 2, 0 }
    };

    // Variable hyperparams
    const std::vector<std::filesystem::path> datafiles{
        "../data/train_test_src/17_measured.csv",
        "../data/train_test_src/22_measured.csv",
        "../data/train_test_src/25_measured.csv",
        "../data/train_test_src/28_measured.csv",
    };
    const std::vector<unsigned>    seeds{ 100 };
    const std::vector<UTIL::Index> res_sizes{ 350, 500 }, warmup_sizes{ 0 };
    const std::vector<T>           leak_rates{ 0.025, 0.05, 0.075, 0.1,  0.15,
                                     0.25,  0.35, 0.55,  0.85, 0.95 },
        sparsity_values{ 0.1, 0.2, 0.3, 0.4 },
        spectral_radii{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 },
        alpha_values{ 1E-2, 1E-5, 1E-7, 1E-10 }, input_scales{ 0.5 };

    const auto N_tests{ datafiles.size() * seeds.size() * res_sizes.size()
                        * warmup_sizes.size() * leak_rates.size()
                        * sparsity_values.size() * spectral_radii.size()
                        * alpha_values.size() * input_scales.size() };

    std::vector<std::string> generated_files;

    std::cout << "Running " << N_tests << " tests.\n";
    auto total_count{ 0 };
    for ( const auto seed : seeds ) {
        for ( const auto & path : datafiles ) {
            // Count to track number of tests for this file & seed
            auto                     count{ 0 };
            std::chrono::duration<T> total_time{ 0 }, total_test_time{ 0 };

            // Load data csv
            const auto data_csv{ CSV::SimpleCSV<T>(
                /*filename*/ path, /*col_titles*/ true, /*skip_header*/ 0,
                /*delim*/ ",", /*max_line_size*/ 256 )

            };
            const auto data_pool{ data_csv.atv( 0, 0 ) };

            // Split into train & test sets
            const auto [train_pair, test_labels] =
                data_split<T>( data_pool, 0.75, feature_shape, data_stride,
                               UTIL::Standardizer<T>{} );
            const auto [train_samples, train_labels] = train_pair;

            // Keeping track of best parameters for current file
            unsigned    best_seed{ 0 };
            UTIL::Index best_res{ 0 }, best_warmup{ 0 };
            T           best_leak{ 0 }, best_sparsity{ 0 }, best_radius{ 0 },
                best_alpha{ 0 }, best_scale{ 0 }, best_RMSE{ 0 };

            nlohmann::json file_json;
            for ( const auto warmup : warmup_sizes ) {
                for ( const auto radius : spectral_radii ) {
                    for ( const auto n_node : res_sizes ) {
                        for ( const auto sparsity : sparsity_values ) {
                            for ( const auto input_scale : input_scales ) {
                                for ( const auto leak : leak_rates ) {
                                    for ( const auto alpha : alpha_values ) {
                                        const auto start{ clock::now() };

                                        std::cout << std::format(
                                            "Running {} / {} tests ({}, seed: "
                                            "{}, "
                                            "file: {})...\n",
                                            total_count, N_tests, count, seed,
                                            path.string() );

                                        // Initialise ESN
                                        const auto init_start{ clock::now() };
                                        ESN::ESN<
                                            T,
                                            ESN::input_t::dense
                                                | ESN::input_t::homogeneous,
                                            ESN::adjacency_t::sparse,
                                            ESN::feature_t::bias
                                                | ESN::feature_t::linear
                                                | ESN::feature_t::reservoir,
                                            UTIL::L2Solver<T>, std::mt19937,
                                            true>
                                            esn(
                                                d, n_node, leak, sparsity,
                                                radius, seed, warmup, bias,
                                                input_scale,
                                                []( const T x ) {
                                                    return std::tanh( x );
                                                },
                                                UTIL::L2Solver<T>( alpha ) );
                                        const auto init_end{ clock::now() };

                                        // Train ESN
                                        const auto train_start{ clock::now() };
                                        esn.train(
                                            train_samples.rightCols( d ),
                                            train_labels.rightCols( d ) );
                                        const auto train_end{ clock::now() };

                                        // Forecast ESN
                                        const auto forecast_start{
                                            clock::now()
                                        };
                                        const auto forecast{ esn.forecast(
                                            test_labels.rightCols( d ),
                                            { 0 } ) };
                                        const auto forecast_end{ clock::now() };

                                        const auto test_time{
                                            std::chrono::duration_cast<
                                                std::chrono::duration<T>>(
                                                ( forecast_end
                                                  - forecast_start )
                                                + ( train_end - train_start )
                                                + ( init_end - init_start ) )
                                        };

                                        total_test_time += test_time;

                                        const auto avg_test_time{
                                            total_time
                                            / static_cast<T>( count + 1 )
                                        };

                                        // Statistics
                                        const auto overall_rmse{ UTIL::RMSE<T>(
                                            forecast,
                                            test_labels.rightCols( d )
                                                .bottomRows( test_labels.rows()
                                                             - warmup ) ) };

                                        std::cout << std::format(
                                            "\t- RMSE: {}\n\t- best_RMSE: {}\n",
                                            overall_rmse( 1 ), best_RMSE );

                                        // Column titles for data output
                                        const std::vector<std::string>
                                            col_titles{ "t",  "I",  "V",
                                                        "I'", "V'", "rmse" };

                                        // Add to file json
                                        file_json[count] = {
                                            { "seed", seed },
                                            { "n_node", n_node },
                                            { "warmup", warmup },
                                            { "leak", leak },
                                            { "sparsity", sparsity },
                                            { "radius", radius },
                                            { "alpha", alpha },
                                            { "scale", input_scale },
                                            { "rmse", overall_rmse( 1 ) },
                                            { "init_time",
                                              std::format(
                                                  "{}",
                                                  std::chrono::duration_cast<
                                                      std::chrono::duration<T>>(
                                                      init_end
                                                      - init_start ) ) },
                                            { "train_time",
                                              std::format(
                                                  "{}",
                                                  std::chrono::duration_cast<
                                                      std::chrono::duration<T>>(
                                                      train_end
                                                      - train_start ) ) },
                                            { "forecast_time",
                                              std::format(
                                                  "{}",
                                                  std::chrono::duration_cast<
                                                      std::chrono::duration<T>>(
                                                      forecast_end
                                                      - forecast_start ) ) }
                                        };

                                        // Keep track of best parameter set so
                                        // far
                                        if ( count == 0 ) {
                                            best_seed = seed;
                                            best_res = n_node;
                                            best_warmup = warmup;
                                            best_leak = leak;
                                            best_sparsity = sparsity;
                                            best_radius = radius;
                                            best_alpha = alpha;
                                            best_scale = input_scale;
                                            best_RMSE = overall_rmse( 1 );
                                        }
                                        else {
                                            if ( overall_rmse( 1 )
                                                 < best_RMSE ) {
                                                best_seed = seed;
                                                best_res = n_node;
                                                best_warmup = warmup;
                                                best_leak = leak;
                                                best_sparsity = sparsity;
                                                best_radius = radius;
                                                best_alpha = alpha;
                                                best_scale = input_scale;
                                                best_RMSE = overall_rmse( 1 );
                                            }
                                        }

                                        const auto end{ clock::now() };

                                        total_time +=
                                            std::chrono::duration_cast<
                                                std::chrono::duration<T>>(
                                                end - start );

                                        const auto avg_time{ total_time
                                                             / static_cast<T>(
                                                                 count + 1 ) };

                                        const auto est_final_time{
                                            total_time
                                            + static_cast<T>( N_tests - count
                                                              - 1 )
                                                  * avg_time
                                        };
                                        const auto remaining_seconds{
                                            est_final_time - total_time
                                        };

                                        std::cout << std::format(
                                            "\t- Average test time: {}.\n\t- "
                                            "Average iteration time: {}.\n\t- "
                                            "Estimated remaining time: {} "
                                            "seconds. Estimated runtime: {}\n",
                                            avg_test_time, avg_time,
                                            remaining_seconds, est_final_time );

                                        count++;
                                        total_count++;
                                        std::cout << "\t- Done.\n";
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Write file_json to metadata folder
            const auto json_filename{ get_filename( std::format(
                "../data/metadata/{}_{}.json", path.stem().string(), seed ) ) };

            std::ofstream json_out( json_filename );
            json_out << std::setw( 4 ) << file_json << std::endl;
            generated_files.push_back( json_filename );

            // Print out best parameters for current dataset
            std::cout << std::format(
                "For {}, the best parameter set found was:\n\t- seed: "
                "{}\n\t- "
                "n_node: {}\n\t- warmup: {}\n\t- leak: {}\n\t- sparsity: "
                "{}\n\t- "
                "radius: {}\n\t- alpha: {}\n\t- scale: {}\n\t- rmse: {}\n",
                path.string(), best_seed, best_res, best_warmup, best_leak,
                best_sparsity, best_radius, best_alpha, best_scale, best_RMSE );
        }
    }

    std::cout << "Generated files:\n";
    for ( const auto & file : generated_files ) {
        std::cout << "\t- " << file << "\n";
    }

    std::vector<std::string> input_files( datafiles.size() );
    std::transform( datafiles.cbegin(), datafiles.cend(), input_files.begin(),
                    []( const auto path ) { return path.string(); } );

    nlohmann::json result_metadata;
    result_metadata["params"] = { { "input_files", input_files },
                                  { "seeds", seeds },
                                  { "N", res_sizes },
                                  { "warmup", warmup_sizes },
                                  { "leak", leak_rates },
                                  { "sparsity", sparsity_values },
                                  { "radius", spectral_radii },
                                  { "alpha", alpha_values },
                                  { "scale", input_scales } };
    result_metadata["output_files"] = generated_files;

#endif
#ifdef DOUBLESCROLL_OPT
#endif
#ifdef RMSE_TEST
    const auto x{ UTIL::Mat<T>::Random( 100, 3 ) };
    const auto dx{ UTIL::Mat<T>::Random( 100, 3 ) };
    const auto dy{ UTIL::Mat<T>::Constant( 100, 3, 2. ) };
    const auto x_{ x + dx };
    const auto y{ x + dy };
    const auto RMSE{ UTIL::RMSE<T>( x_, x ) };
    const auto RMSE_{ UTIL::RMSE<T>( x, y ) };

    std::cout << "RMSE:\n" << RMSE << std::endl;
    std::cout << "RMSE_:\n" << RMSE_ << std::endl;

    const auto window_RMSE_10{ UTIL::windowed_RMSE<T>( x, y, 1 ) };
    std::cout << std::endl;
    const auto window_RMSE_11{ UTIL::windowed_RMSE<T>( x, y, 11 ) };

    std::cout << "window_RMSE_10:\n" << window_RMSE_10 << std::endl;
    std::cout << "window_RMSE_11:\n" << window_RMSE_11 << std::endl;
#endif
#ifdef ESN_TEST
    using T = double;

    const std::filesystem::path data_path{
        "../data/train_test_src/17_measured.csv"
    };
    const auto data_csv{ CSV::SimpleCSV<T>(
        /*filename*/ data_path, /*col_titles*/ true, /*skip_header*/ 0,
        /*delim*/ ",", /*max_line_size*/ 256 ) };
    const auto data_pool{ data_csv.atv( 0, 0 ) };

    const unsigned    seed{ 1284 };
    const UTIL::Index d{ 2 }, n_node{ 4000 }, n_warmup{ 000 }, data_stride{ 2 };
    const T           leak{ 0.85 }, sparsity{ 0.005 }, spectral_radius{ 0.8 },
        alpha{ 4E-3 }, bias{ 1. }, input_scale{ 1. };

    const auto activation_func = []<UTIL::Weight T>( const T x ) {
        return std::tanh( x );
    };
    const auto solver = UTIL::L2Solver<T>( T{ alpha } );
    const auto w_in_func =
        []<UTIL::Weight T, UTIL::RandomNumberEngine Generator>(
            [[maybe_unused]] const T x, [[maybe_unused]] Generator & gen ) {
            static auto dist{ std::uniform_real_distribution<T>( -1., 1. ) };
            return dist( gen );
        };
    const auto adjacency_func =
        []<UTIL::Weight T, UTIL::RandomNumberEngine Generator>(
            [[maybe_unused]] const T x, [[maybe_unused]] Generator & gen ) {
            static auto dist{ std::uniform_real_distribution<T>( -1., 1. ) };
            return dist( gen );
        };

    std::cout << std::format(
        "d: {}, n_node: {}, leak: {}, sparsity: {}, spectral_radius: {}, seed: "
        "{}, n_warmup: {}\n",
        d, n_node, leak, sparsity, spectral_radius, seed, n_warmup );
    std::cout << std::format( "data_stride: {}\n", data_stride );

    const std::vector<std::tuple<UTIL::Index, UTIL::Index>> feature_shape{
        { 0, 0 }, { 1, 0 }, { 2, 0 }
    };

    const auto [train_pair, test_labels] = data_split<T>(
        data_pool, 0.75, feature_shape, data_stride, UTIL::Standardizer<T>{} );
    const auto [train_samples, train_labels] = train_pair;

    std::cout << std::format( "train_samples: {}, train_labels: {}\n",
                              shape_str<T, -1, -1>( train_samples ),
                              shape_str<T, -1, -1>( train_labels ) );

    std::cout << "Init..." << std::endl;
    const auto init_start{ std::chrono::steady_clock::now() };
    ESN::ESN<
        /* value_t */ T,
        /* input weight type */ ESN::input_t::dense | ESN::input_t::homogeneous,
        /* adjacency matrix type */ ESN::adjacency_t::sparse,
        /* feature_shape */ ESN::feature_t::bias | ESN::feature_t::linear
            | ESN::feature_t::reservoir,
        /* Solver type */ UTIL::L2Solver<T>,
        /* generator type */ std::mt19937,
        /* target_difference */ true>
        split_dense( d, n_node, leak, sparsity, spectral_radius, seed, n_warmup,
                     bias, input_scale, { 1 }, activation_func, solver,
                     w_in_func, adjacency_func );

    const auto init_finish{ std::chrono::steady_clock::now() };

    std::cout << "Train..." << std::endl;
    const auto train_start{ std::chrono::steady_clock::now() };
    split_dense.train( train_samples.rightCols( d ),
                       train_labels.rightCols( d ) );

    const auto train_finish{ std::chrono::steady_clock::now() };

    std::cout << "Forecast..." << std::endl;
    const auto forecast_start{ std::chrono::steady_clock::now() };
    const auto forecast{ split_dense.forecast( test_labels.rightCols( d ) ) };
    const auto forecast_finish{ std::chrono::steady_clock::now() };

    const auto rmse{ UTIL::RMSE<T>( forecast,
                                    test_labels.rightCols( d ).bottomRows(
                                        test_labels.rows() - n_warmup ) ) };

    std::cout << std::format( "RMSE: {}\n", rmse( 1 ) );

    std::cout << std::format(
        "Initialising ESN took: {}\n",
        std::chrono::duration_cast<std::chrono::duration<T>>( init_finish
                                                              - init_start ) );
    std::cout << std::format(
        "Training ESN took: {}\n",
        std::chrono::duration_cast<std::chrono::duration<T>>( train_finish
                                                              - train_start ) );
    std::cout << std::format(
        "Forecasting ESN took: {}\n",
        std::chrono::duration_cast<std::chrono::duration<T>>(
            forecast_finish - forecast_start ) );

    // Write results to file
    // Write location
    const std::filesystem::path write_path{
        "../data/forecast_data/esn_forecast.csv"
    };

    const std::vector<std::string> col_titles{ "t", "I", "V", "I'", "V'" };

    UTIL::Mat<T> forecast_data( forecast.rows(),
                                static_cast<UTIL::Index>( col_titles.size() ) );

    std::cout << std::format( "test_labels: {}, forecast: {}\n",
                              UTIL::mat_shape_str<T, -1, -1>( test_labels ),
                              UTIL::mat_shape_str<T, -1, -1>( forecast ) );

    forecast_data << test_labels.leftCols( 1 ).bottomRows( test_labels.rows()
                                                           - n_warmup ),
        forecast,
        test_labels.rightCols( d ).bottomRows( test_labels.rows() - n_warmup );

    // Write to a csv
    const auto write_success{ CSV::SimpleCSV<T>::template write<T>(
        write_path, forecast_data, col_titles ) };

    if ( !write_success ) {
        std::cerr << std::format( "Unable to write forecast data.\n" );

        return EXIT_FAILURE;
    }
#endif
#ifdef ESN_DOUBLESCROLL
    using T = double;

    // Doublescroll
    const std::filesystem::path doublescroll_train_path{
        "../data/train_data/doublescroll.csv"
    };
    const std::filesystem::path doublescroll_test_path{
        "../data/test_data/doublescroll.csv"
    };
    // clang-format off
    const auto doublescroll_train_csv{ CSV::SimpleCSV<T>(
        /*filename=*/doublescroll_train_path,
        /*col_titles=*/true,
        /*skip_header=*/0,
        /*delim*/ ",",
        /*max_line_size=*/256 ) };
    const auto doublescroll_test_csv{ CSV::SimpleCSV<T>(
        /*filename=*/doublescroll_test_path,
        /*col_titles=*/true,
        /*skip_header=*/0,
        /*delim*/ ",",
        /*max_line_size=*/256 ) };
    // clang-format on

    const auto doublescroll_train_data{ doublescroll_train_csv.atv() };
    const auto doublescroll_test_data{ doublescroll_test_csv.atv() };
    std::cout << std::format(
        "train_data: {}\ntest_data: {}\n",
        UTIL::mat_shape_str<T, -1, -1>( doublescroll_train_data ),
        UTIL::mat_shape_str<T, -1, -1>( doublescroll_test_data ) );

    const UTIL::FeatureVecShape doublescroll_feature_shape{
        { 0, 0 }, { 1, 0 }, { 2, 0 }, { 3, 0 }
    };

    const unsigned    seed{ 423 };
    const UTIL::Index d{ 3 }, n_node{ 1500 }, n_warmup{ 0 }, data_stride{ 1 };
    const T           leak{ 0.05 }, sparsity{ 0.005 }, spectral_radius{ 0.25 },
        alpha{ 8E-6 }, bias{ 1. }, input_scale{ 1. };
    const std::vector<UTIL::Index> train_targets{ 1 };

    const auto activation_func = []<UTIL::Weight T>( const T x ) {
        return std::tanh( x );
    };
    const auto solver = UTIL::L2Solver<T>( T{ alpha } );
    const auto w_in_func =
        []<UTIL::Weight T, UTIL::RandomNumberEngine Generator>(
            [[maybe_unused]] const T x, [[maybe_unused]] Generator & gen ) {
            static auto dist{ std::uniform_real_distribution<T>( -1., 1. ) };
            return dist( gen );
        };
    const auto adjacency_func =
        []<UTIL::Weight T, UTIL::RandomNumberEngine Generator>(
            [[maybe_unused]] const T x, [[maybe_unused]] Generator & gen ) {
            static auto dist{ std::uniform_real_distribution<T>( -1., 1. ) };
            return dist( gen );
        };

    std::cout << std::format(
        "d: {}, n_node: {}, leak: {}, sparsity: {}, spectral_radius: {}, seed: "
        "{}, n_warmup: {}\n",
        d, n_node, leak, sparsity, spectral_radius, seed, n_warmup );
    std::cout << std::format( "data_stride: {}\n", data_stride );

    const auto init_start{ std::chrono::steady_clock::now() };
    ESN::ESN<
        /* value_t */ T,
        /* input weight type */ ESN::input_t::dense | ESN::input_t::homogeneous,
        /* adjacency matrix type */ ESN::adjacency_t::sparse,
        /* feature_shape */ ESN::feature_t::bias | ESN::feature_t::linear
            | ESN::feature_t::reservoir,
        /* Solver type */ UTIL::L2Solver<T>,
        /* generator type */ std::mt19937,
        /* target_difference */ true>
        split_dense( d, n_node, leak, sparsity, spectral_radius, seed, n_warmup,
                     bias, input_scale, train_targets, activation_func, solver,
                     w_in_func, adjacency_func );

    const auto init_finish{ std::chrono::steady_clock::now() };

    const auto [doublescroll_train_pair, doublescroll_test_labels] =
        data_split<T>( doublescroll_train_data, doublescroll_test_data,
                       doublescroll_feature_shape, data_stride,
                       UTIL::Standardizer<T>{} );
    const auto [doublescroll_train_samples, doublescroll_train_labels] =
        doublescroll_train_pair;

    const auto train_start{ std::chrono::steady_clock::now() };
    split_dense.train( doublescroll_train_samples.rightCols( d ),
                       doublescroll_train_labels.rightCols( d ) );

    const auto train_finish{ std::chrono::steady_clock::now() };

    const auto forecast_start{ std::chrono::steady_clock::now() };

    const auto forecast{ split_dense.forecast(
        doublescroll_test_labels.rightCols( d ) ) };

    const auto forecast_finish{ std::chrono::steady_clock::now() };

    std::cout << std::format(
        "Initialising ESN took: {}\n",
        std::chrono::duration_cast<std::chrono::duration<T>>( init_finish
                                                              - init_start ) );
    std::cout << std::format(
        "Training ESN took: {}\n",
        std::chrono::duration_cast<std::chrono::duration<T>>( train_finish
                                                              - train_start ) );
    std::cout << std::format(
        "Forecasting ESN took: {}\n",
        std::chrono::duration_cast<std::chrono::duration<T>>(
            forecast_finish - forecast_start ) );

    // Write results to file
    // Write location
    const std::filesystem::path write_path{
        "../data/forecast_data/esn_doublescroll_forecast.csv"
    };

    const std::vector<std::string> col_titles{ "t",  "I",   "V1", "V2",
                                               "I'", "V1'", "V2'" };

    std::cout << std::format(
        "forecast: {}\ndoublescroll_test_labels: {}\n",
        UTIL::mat_shape_str<T, -1, -1>( forecast ),
        UTIL::mat_shape_str<T, -1, -1>( doublescroll_test_labels ) );
    UTIL::Mat<T> forecast_data(
        forecast.rows(), forecast.cols() + doublescroll_test_labels.cols() );

    forecast_data << doublescroll_test_labels.leftCols( 1 ).bottomRows(
        doublescroll_test_labels.rows() - n_warmup ),
        forecast,
        doublescroll_test_labels.rightCols( d ).bottomRows(
            doublescroll_test_labels.rows() - n_warmup );

    // Write to a csv
    const auto write_success{ CSV::SimpleCSV<T>::template write<T>(
        write_path, forecast_data, col_titles ) };

    if ( !write_success ) {
        std::cerr << std::format( "Unable to write forecast data.\n" );

        return EXIT_FAILURE;
    }
#endif
#ifdef FORECAST
    using T = double;

    const std::filesystem::path data_path{
        "../data/train_test_src/17_measured.csv"
    };
    const auto data_csv{ CSV::SimpleCSV<T>(
        /*filename*/ data_path, /*col_titles*/ true, /*skip_header*/ 0,
        /*delim*/ ",", /*max_line_size*/ 256 ) };
    const auto data_pool{ data_csv.atv( 0, 0 ) };

    const std::vector<std::tuple<UTIL::Index, UTIL::Index>> feature_shape{
        { 0, 0 }, { 1, 0 }, { 2, 0 }
    };
    const bool                 use_const{ true };
    const T /*alpha{ 1E-4 },*/ constant{ 1 };
    const UTIL::Index d{ 2 }, /*k{ 1 }, s{ 1 }, p{ 3 },*/ data_stride{ 4 };
    const std::vector<UTIL::Index> train_targets{ 1, 2, 3 };

    const std::vector<UTIL::Index> kvals{ 1, 2, 3 },
        svals{ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
        pvals{ 2, 3, 4 };
    const std::vector<T> alphavals{ 10, 1, .1, .01, .001, 0.0001, 0.00001 };

    UTIL::Index best_k{ 1 }, best_s{ 1 }, best_p{ 1 }, best_alpha{ 10 };
    T           best_rmse{ 1000000000000. };

    for ( const auto alpha : alphavals ) {
        for ( const auto k : kvals ) {
            for ( const auto s : svals ) {
                for ( const auto p : pvals ) {
                    std::cout << std::format(
                        "d = {}, k = {}, s = {}, p = {}, alpha = {}\n", d, k, s,
                        p, alpha );

                    const auto [train_pair, test_pair] =
                        data_split<T>( data_pool, 0.75, feature_shape,
                                       NVAR::warmup_offset( k, s ), data_stride,
                                       UTIL::Standardizer<T>{} );
                    const auto [train_samples, train_labels] = train_pair;
                    const auto [test_warmup, test_labels] = test_pair;

                    NVAR::NVAR<T, NVAR::nonlinear_t::poly> test(
                        train_samples.rightCols( d ),
                        train_labels.rightCols( d ), d, k, s, p, use_const,
                        constant, train_targets, UTIL::L2Solver<T>( alpha ),
                        false );


                    const auto forecast{ test.forecast(
                        test_warmup.rightCols( d ),
                        test_labels.rightCols( d ) ) };

                    const auto rmse{ UTIL::RMSE<T>(
                        forecast, test_labels.rightCols( d ) ) };
                    std::cout << "rmse: " << rmse << std::endl;
                    if ( rmse( 1 ) < best_rmse ) {
                        best_rmse = rmse( 1 );
                        best_k = k;
                        best_s = s;
                        best_p = p;
                        best_alpha = alpha;
                    }
                }
            }
        }
    }

    const auto [train_pair, test_pair] =
        data_split<T>( data_pool, 0.75, feature_shape,
                       NVAR::warmup_offset( best_k, best_s /*k, s*/ ),
                       data_stride, UTIL::Standardizer<T>{} );
    const auto [train_samples, train_labels] = train_pair;
    const auto [test_warmup, test_labels] = test_pair;

    NVAR::NVAR<T, NVAR::nonlinear_t::poly> test(
        train_samples.rightCols( d ), train_labels.rightCols( d ), d, best_k,
        best_s, best_p, /*k, s, p,*/ use_const, constant, train_targets,
        UTIL::L2Solver<T>( best_alpha /*alpha*/ ), true, { "I", "V" },
        "../data/forecast_data/tmp.csv" );

    const auto forecast{ test.forecast( test_warmup.rightCols( d ),
                                        test_labels.rightCols( d ) ) };

    const auto rmse{ UTIL::RMSE<T>( forecast, test_labels.rightCols( d ) ) };
    std::cout << "RMSE: " << rmse << std::endl;

    std::cout << std::format(
        "best_rmse: {}, best_k: {}, best_s: {}, best_p: {}, best_alpha: {}\n",
        best_rmse, best_k, best_s, best_p, best_alpha );

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
        forecast_col_titles.size(), test_labels.cols(), forecast.cols(),
        test_labels.cols() );
    UTIL::Mat<T> forecast_data( forecast.rows(),
                                forecast.cols() + test_labels.cols() );

    forecast_data << test_labels.leftCols( 1 ), forecast,
        test_labels.rightCols( d );

    const auto write_success{ CSV::SimpleCSV<T>::template write<T>(
        forecast_path, forecast_data, forecast_col_titles ) };

    if ( !write_success ) {
        std::cerr << std::format( "Unable to write forecast data.\n" );
    }

#endif
#ifdef CUSTOM_FEATURES
    using T = float;

    const std::filesystem::path data_path{
        "../data/train_test_src/22_measured.csv"
    };
    const auto data_csv{ CSV::SimpleCSV<T>(
        /*filename*/ data_path, /*col_titles*/ true, /*skip_header*/ 0,
        /*delim*/ ",", /*max_line_size*/ 256 ) };
    const auto data_pool{ data_csv.atv( 0, 0 ) };

    const bool   use_const{ true };
    const double alpha{ 0 }, constant{ 1 };
    const Index  d{ 4 }, k{ 2 }, s{ 10 }, p{ 3 }, data_stride{ 4 }, delay{ 1 };
    std::cout << std::format( "data_stride: {}, delay: {}\n", data_stride,
                              delay );
    std::cout << std::format( "alpha: {}, use_const: {}, constant: {}\n", alpha,
                              use_const ? "true" : "false", constant );
    std::cout << std::format( "d = {}, k = {}, s = {}, p = {}\n", d, k, s, p );

    // Feature shape of t_(n), V_(n), V_(n-1), I_(n), I_(n-1)
    const FeatureVecShape feature_shape{
        { 0, 0 }, { 2, 0 }, { 2, delay }, { 1, 0 }, { 1, delay }
    };

    // Get train / test data
    const auto [train_pair, test_pair] = data_split<double>(
        data_pool, 0.75, feature_shape, NVAR::warmup_offset( k, s ),
        data_stride, UTIL::NullProcessor<double>{} );
    const auto [train_samples, train_labels] = train_pair;
    const auto [test_warmup, test_labels] = test_pair;

    // Building NVAR
    std::cout << "Building NVAR.\n";
    NVAR::NVAR<double, NVAR::nonlinear_t::poly> test(
        train_samples.rightCols( d ), train_labels.rightCols( d ), d, k, s, p,
        use_const, double{ constant }, L2Solver( alpha ), true,
        { "V_(n)", "V_(n-1)", "I_(n)", "I_(n-1)" },
        "../data/forecast_data/tmp.csv" );

    std::cout << "Forecasting.\n";
    auto forecast{ test.forecast( test_warmup.rightCols( d ),
                                  test_labels.rightCols( d ),
                                  std::vector<Index>{ 2, 3 } ) };

    // Write results out
    std::cout << "Writing results.\n";
    const std::filesystem::path forecast_path{
        "../data/forecast_data/forecast.csv"
    };
    const std::vector<std::string> forecast_col_titles{
        "t",      "V_(n)",    "V_(n-1)", "I_(n)",   "I_(n-1)",
        "V_(n)'", "V_(n-1)'", "I_(n)'",  "I_(n-1)'"
    };
    std::cout << std::format(
        "forecast_col_titles.size(): {},  forecast.cols(): "
        "{}, test_labels.cols(): {}\n",
        forecast_col_titles.size(), forecast.cols(), test_labels.cols() );
    Mat<double> forecast_data( forecast.rows(),
                               forecast.cols() + test_labels.cols() );

    forecast_data << test_labels.leftCols( 1 ), forecast,
        test_labels.rightCols( d );

    const auto write_success{ CSV::SimpleCSV<T>::write<T>(
        forecast_path, forecast_data, forecast_col_titles ) };

    if ( !write_success ) {
        std::cerr << std::format( "Unable to write forecast data.\n" );
    }

#endif
#ifdef DOUBLESCROLL
    using T = double;

    std::cout << "Running doublescroll." << std::endl;

    // Doublescroll
    const std::filesystem::path doublescroll_train_path{
        "../data/train_data/doublescroll.csv"
    };
    const std::filesystem::path doublescroll_test_path{
        "../data/test_data/doublescroll.csv"
    };
    // clang-format off
    const auto doublescroll_train_csv{ CSV::SimpleCSV<T>(
        /*filename=*/doublescroll_train_path,
        /*col_titles=*/true,
        /*skip_header=*/0,
        /*delim*/ ",",
        /*max_line_size=*/256 ) };
    const auto doublescroll_test_csv{ CSV::SimpleCSV<T>(
        /*filename=*/doublescroll_test_path,
        /*col_titles=*/true,
        /*skip_header=*/0,
        /*delim*/ ",",
        /*max_line_size=*/256 ) };
    // clang-format on

    const auto doublescroll_train_data{ doublescroll_train_csv.atv() };
    const auto doublescroll_test_data{ doublescroll_test_csv.atv() };

    const bool                     use_const{ false };
    const UTIL::Index              d2{ 3 }, k2{ 2 }, s2{ 2 }, p2{ 3 };
    const T                        alpha{ 1E-1 }, constant{ 1 };
    const std::vector<UTIL::Index> targets{ 2 };
    std::cout << std::format(
        "doublescroll_train_data: {}\n",
        UTIL::mat_shape_str<double, -1, -1>( doublescroll_train_data ) );
    const UTIL::FeatureVecShape feature_shape{
        { 0, 0 }, { 1, 0 }, { 2, 0 }, { 3, 0 }
    };

    const auto [doublescroll_train_pair, doublescroll_test_pair] =
        data_split<double>( doublescroll_train_data, doublescroll_test_data,
                            feature_shape, NVAR::warmup_offset( k2, s2 ), 1,
                            UTIL::Standardizer<T>{} );
    const auto [doublescroll_train_samples, doublescroll_train_labels] =
        doublescroll_train_pair;
    const auto [doublescroll_warmup, doublescroll_test_labels] =
        doublescroll_test_pair;

    // Create NVAR model
    NVAR::NVAR<T, NVAR::nonlinear_t::poly, UTIL::L2Solver<T>, false>
        doublescroll_test(
            doublescroll_train_samples.rightCols( d2 ),
            doublescroll_train_labels.rightCols( d2 ), d2, k2, s2, p2,
            use_const, constant, targets, UTIL::L2Solver<T>( alpha ), true,
            { "v1", "v2", "I" },
            "../data/forecast_data/doublescroll_reconstruct.csv" );

    // Forecast
    auto doublescroll_forecast{ doublescroll_test.forecast(
        doublescroll_warmup.rightCols( d2 ),
        doublescroll_test_labels.rightCols( d2 ) ) };

    std::cout << "RMSE:\n"
              << UTIL::RMSE<T>( doublescroll_forecast,
                                doublescroll_test_labels.rightCols( d2 ) )
              << std::endl;

    // Write data
    const std::vector<std::string> col_titles{ "t",   "v1",  "v2", "I",
                                               "v1'", "v2'", "I'" };
    UTIL::Mat<T>                   write_data( doublescroll_forecast.rows(),
                                               static_cast<UTIL::Index>( col_titles.size() ) );
    write_data << doublescroll_test_labels.leftCols( 1 ), doublescroll_forecast,
        doublescroll_test_labels.rightCols( d2 );

    const std::filesystem::path doublescroll_forecast_path{
        "../data/forecast_data/"
        "doublescroll_predict.csv"
    };

    const auto write_success{ CSV::SimpleCSV<T>::write<T>(
        doublescroll_forecast_path, write_data, col_titles ) };

    if ( !write_success ) {
        std::cerr << std::format( "Unable to write forecast data.\n" );
    }
#endif
#ifdef HH_MODEL
    using T = float;

    const auto base_path{ std::filesystem::absolute(
        std::filesystem::current_path() / ".." ) };

    const std::vector<std::string> files{ "a2t11", "a4t15", "a6t12", "a6t38",
                                          "a8t19" };
    std::vector<std::string>       results_files;

    const bool   use_const{ true };
    const double alpha{ 0.1 }, constant{ 1 };
    const Index  d{ 2 }, k{ 5 }, s{ 20 }, p{ 5 }, data_stride{ 5 }, delay{ 1 };
    std::cout << std::format( "data_stride: {}, delay: {}\n", data_stride,
                              delay );
    std::cout << std::format( "alpha: {}, use_const: {}, constant: {}\n", alpha,
                              use_const ? "true" : "false", constant );
    std::cout << std::format( "d = {}, k = {}, s = {}, p = {}\n", d, k, s, p );

    for ( const auto & file : files ) {
        const auto python_file = base_path / "data" / "decompress_signal.py";
        const auto read_path = base_path / "data" / "hh_data" / file;
        const auto write_path = base_path / "tmp" / ( file + ".csv" );

        std::cout << "Reading from: " << read_path << std::endl;
        std::cout << "Writing to: " << write_path << std::endl;

        std::cout << "Decompressing..." << std::endl;
        const std::string decompress_cmd{ std::format(
            "python3.11 {} {} {}", python_file.string(), read_path.string(),
            write_path.string() ) };
        const auto [decompress_result_code, decompress_msg] =
            exec( decompress_cmd.c_str() );

        std::cout << std::format(
            "{}:\nResult code: {}\n-----\nMessage:\n-----\n{}\nDone.\n", file,
            decompress_result_code, decompress_msg );

        if ( decompress_result_code == 0 /* Success code */ ) {
            std::cout << "Successful decompress" << std::endl;
            // Load data
            std::cout << "Loading csv data." << std::endl;
            // clang-format off
            const auto csv{ CSV::SimpleCSV<T>(
                /*filename=*/write_path,
                /*col_titles=*/true,
                /*skip_header=*/0,
                /*delim*/ ",",
                /*max_line_size=*/256 )
            };
            // clang-format on
            std::cout << "Loading csv into matrix." << std::endl;
            const Mat<double> full_data{ csv.atv( 0, 0 ) };

            // Split data
            std::cout << "Splitting data..." << std::endl;

            const FeatureVecShape shape{ { 0, 0 }, { 1, 0 }, { 2, 0 } };

            const auto [train_pair, test_pair] = data_split<double>(
                /* data */ full_data, /* train_test_ratio */ 0.75,
                /* shape */ shape,
                /*warmup_offset*/NVAR::warmup_offset/* k */ k,
                /* s */ s), /* stride */ data_stride, /* DataProcessor */ UTIL::NullProcessor<double>{} );

            const auto [train_samples, train_labels] = train_pair;
            const auto [test_warmup, test_labels] = test_pair;

            std::cout << std::format(
                "train_samples: {}\ntrain_labels: {}\ntest_warmup: "
                "{}\ntest_labels: {}\n",
                UTIL::mat_shape_str<double, -1, -1>( train_samples ),
                UTIL::mat_shape_str<double, -1, -1>( train_labels ),
                UTIL::mat_shape_str<double, -1, -1>( test_warmup ),
                UTIL::mat_shape_str<double, -1, -1>( test_labels ) );

            std::cout << "Done." << std::endl;

            // Train NVAR
            std::cout << "Training NVAR..." << std::endl;

            NVAR::NVAR<double> nvar(
                train_samples.rightCols( d ), train_labels.rightCols( d ), d, k,
                s, p, use_const, constant, L2Solver( alpha ), true,
                { "Vmembrane", "Istim" }, "../data/forecast_data/tmp.csv" );

            std::cout << "Done." << std::endl;

            // Forecast forwards
            std::cout << "Forecasting..." << std::endl;

            const auto forecast{ nvar.forecast( test_warmup.rightCols( d ),
                                                test_labels.rightCols( d ),
                                                { 1 } ) };

            std::cout << "Done." << std::endl;

            // Write result file
            std::cout << "Writing result..." << std::endl;
            std::cout << std::format(
                "forecast: {}, test_labels: {}\n",
                UTIL::mat_shape_str<double, -1, -1>( forecast ),
                UTIL::mat_shape_str<double, -1, -1>( test_labels ) );

            const std::filesystem::path forecast_path{
                "../data/forecast_data"
            };
            std::cout << std::format( "forecast_path: {}\n",
                                      forecast_path.string() );
            std::cout << std::format( "file: {}, (file+\".csv\"): {}\n", file,
                                      ( file + ".csv" ) );
            const auto write_file{ forecast_path / ( file + ".csv" ) };
            std::cout << std::format( "write_file: {}\n", write_file.string() );

            const std::vector<std::string> col_titles{ "t", "Vmembrane'",
                                                       "Istim'", "Vmembrane",
                                                       "Istim" };

            std::cout << std::format( "col_titles: {}", col_titles.size() );
            std::cout << "col_titles = " << col_titles << std::endl;

            Mat<double> results( forecast.rows(),
                                 test_labels.cols() + forecast.cols() );

            std::cout << std::format(
                "test_labels: {}, forecast: {}\n",
                UTIL::mat_shape_str<double, -1, -1>( test_labels ),
                UTIL::mat_shape_str<double, -1, -1>( forecast ) );
            std::cout << std::format(
                "results: {}\n",
                UTIL::mat_shape_str<double, -1, -1>( results ) );

            results << test_labels, forecast;

            const auto write_success{ CSV::SimpleCSV<T>::write<T>(
                write_file, results, col_titles ) };

            if ( !write_success ) {
                std::cerr << std::format( "Unable to write forecast data.\n" );
            }
            else {
                results_files.push_back( write_file.string() );
            }

            std::cout << "Done." << std::endl;
        }

        if ( std::filesystem::directory_entry( write_path ).exists() ) {
            // Delete decompressed file
            std::cout << "Deleting decompressed file..." << std::endl;

            const std::string remove_cmd{ std::format( "rm {}",
                                                       write_path.string() ) };

            const auto [result_code, rm_std_output] =
                exec( remove_cmd.c_str() );

            if ( result_code != 0 ) {
                std::cout << "Failed to remove decompressed file." << std::endl;
                std::cout << "Message:\n" << rm_std_output << "\n";
                exit( EXIT_FAILURE );
            }
            std::cout << "Message:\n-----\n" << rm_std_output << "\n-----\n";

            std::cout << "Done.\n" << std::endl;
        }
    }

    /*nlohmann::json file_json;
    file_json["results_files"] = results_files;

    const std::filesystem::path results_file_path{
        "../metadata/results_files.json"
    };
    std::ofstream output( results_file_path );
    output << std::setw( 4 ) << file_json << std::endl;*/

#endif
}
