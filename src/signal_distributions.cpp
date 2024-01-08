#include "CSV/simple_csv.hpp"
#include "util/common.hpp"

#include <filesystem>
#include <string_view>

template <UTIL::Weight T>
UTIL::Mat<T>
joe_change( const UTIL::ConstRefMat<T> & data ) {
    UTIL::Mat<T> result( data.rows(), data.cols() );
    result( Eigen::placeholders::all, std::vector<UTIL::Index>{ 0, 2 } );
    const auto mean{ data.col( 1 ).mean() };
    const auto std{ std::sqrt(
        ( data.col( 1 ).array() - mean )
            .unaryExpr( []( const auto x ) { return x * x; } )
            .sum()
        / static_cast<T>( data.rows() - 1 ) ) };
    result.col( 1 ) = data.col( 1 ) / std;
    return result;
}

int
main() {
    using T = double;

    const auto data_loc{ std::filesystem::absolute(
        std::filesystem::current_path() / ".." / "data" ) };
    const auto src_loc{ data_loc / "train_test_src" };
    const auto write_loc{ data_loc / "distribution_data" };

    const std::vector<std::string_view> data_type{ "measured", "integrated" };
    const std::vector<std::string_view> protocols{ "17", "21", "23" };

    for ( const auto protocol : protocols ) {
        for ( const auto type : data_type ) {
            const auto src_path{ src_loc
                                 / std::format( "{}_{}.csv", protocol, type ) };

            const auto data_csv{ CSV::SimpleCSV<T>( src_path, true, 0, ",",
                                                    256 ) };
            const auto data{ data_csv.atv() };

            [[maybe_unused]] const auto default_result{
                CSV::SimpleCSV<T>::write<T>(
                    write_loc
                        / std::format( "{}_{}_default.csv", protocol, type ),
                    data, data_csv.col_titles() )
            };

            [[maybe_unused]] const auto standardized_result{
                CSV::SimpleCSV<T>::write<T>(
                    write_loc
                        / std::format( "{}_{}_standardized.csv", protocol,
                                       type ),
                    UTIL::Standardizer<T>().pre_process( data, true ),
                    data_csv.col_titles() )
            };

            [[maybe_unused]] const auto normalized_result{
                CSV::SimpleCSV<T>::write<T>(
                    write_loc
                        / std::format( "{}_{}_normalized.csv", protocol, type ),
                    UTIL::Normalizer<T>().pre_process( data, true ),
                    data_csv.col_titles() )
            };

            [[maybe_unused]] const auto joe_result{ CSV::SimpleCSV<T>::write<T>(
                write_loc
                    / std::format( "{}_{}_custom_standardized.csv", protocol,
                                   type ),
                joe_change<T>( data ), data_csv.col_titles() ) };
        }
    }
}
