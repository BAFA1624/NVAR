#include "NVAR_util.hpp"

#include <filesystem>
#include <format>

namespace NVAR
{

std::map<std::string, Index>
parse_filename( const std::string_view filename ) {
    const std::string re{
        "([\\d]+)_([01])_([-\\d]+)_([\\d+]+)_([01])_([\\d]+)_([\\d]+)\\.csv"
    };
    const std::regex                                     pattern{ re };
    std::match_results<std::string_view::const_iterator> match;
    std::map<std::string, Index>                         result = {
        { "N", -1 },       { "train_test", -1 },          { "n_points", -1 },
        { "seed", -1 },    { "measured_integrated", -1 }, { "no", -1 },
        { "total_no", -1 }
    };
    if ( std::regex_search( filename.cbegin(), filename.cend(), match,
                            pattern ) ) {
        result["N"] = std::stol( match[0].str() );
        result["train_test"] = std::stol( match[1].str() );
        result["n_points"] = std::stol( match[2].str() );
        result["seed"] = std::stol( match[3].str() );
        result["measured_integrated"] = std::stol( match[4].str() );
        result["no"] = std::stol( match[5].str() );
        result["total_no"] = std::stol( match[6].str() );
    }

    return result;
}

std::filesystem::path
get_filename( const std::vector<Index> & params ) {
    std::filesystem::path path{};

    switch ( params[1] ) {
    case 0: {
        path += "train_data";
    } break;
    case 1: {
        path += "test_data";
    } break;
    default: {
        path += "forecast_data";
    } break;
    }

    auto params_joined = params | std::views::transform( []( const Index x ) {
                             return std::to_string( x );
                         } )
                         | std::views::join_with( '_' );

    std::string filename{};
    for ( const auto c : params_joined ) { filename += c; }

    return path /= ( filename + ".csv" );
}

std::filesystem::path
get_filename( const std::map<std::string, Index> & file_params ) {
    const auto N{ file_params.at( "N" ) };
    const auto train_test{ file_params.at( "train_test" ) };
    const auto n_points{ file_params.at( "n_points" ) };
    const auto seed{ file_params.at( "seed" ) };
    const auto measured_integrated{ file_params.at( "measured_integrated" ) };
    const auto no{ file_params.at( "no" ) };
    const auto total_no{ file_params.at( "total_no" ) };
    return get_filename( std::vector<Index>{
        N, train_test, n_points, seed, measured_integrated, no, total_no } );
}



}; // namespace NVAR
