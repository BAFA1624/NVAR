#pragma once

#include "NVAR_util.hpp"
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "nlohmann/json.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace NVAR
{

struct TestControlConfig
{
    // Source location for test configs
    std::filesystem::path m_config_source; // Tick
    // Source locations of test/train data
    std::filesystem::path m_train_source; // Tick
    std::filesystem::path m_test_source;  // Tick
    // Output data locations
    std::filesystem::path m_forecast_output;
    std::filesystem::path m_metadata_output;
    // List of test config files to run & if they exist in the filesystem at
    // that location
    std::map<std::string, bool> m_test_files;
    // Flag for successful load
    bool m_successful_load;

    // Retrieval
    [[nodiscard]] inline auto config_source() const noexcept {
        return m_config_source;
    }
    [[nodiscard]] inline auto train_source() const noexcept {
        return m_train_source;
    }
    [[nodiscard]] inline auto test_source() const noexcept {
        return m_test_source;
    }
    [[nodiscard]] inline auto forecast_output() const noexcept {
        return m_forecast_output;
    }
    [[nodiscard]] inline auto metadata_output() const noexcept {
        return m_metadata_output;
    }
    [[nodiscard]] constexpr inline auto & test_files() const noexcept {
        return m_test_files;
    }
    [[nodiscard]] constexpr inline auto success() const noexcept {
        return m_successful_load;
    }
    [[nodiscard]] constexpr inline auto n_test_files() const noexcept {
        return m_test_files.size();
    }

    TestControlConfig(
        const std::filesystem::path & test_control_config_path ) :
        m_successful_load( true ) {
        const std::filesystem::directory_entry test_control_config_entry{
            test_control_config_path
        };

        if ( test_control_config_entry.exists() ) {
            std::ifstream test_control_config( test_control_config_path,
                                               std::ios_base::binary );

            const nlohmann::json json{ nlohmann::json::parse(
                test_control_config ) };

            try {
                m_config_source =
                    std::filesystem::path{ json.at( "config_source" ) };
            }
            catch ( const nlohmann::json::parse_error & e ) {
                std::cerr << std::format(
                    "No config_source listed in test control config: {}\n{}",
                    test_control_config_path.string(), e.what() )
                          << std::endl;
                m_successful_load = false;
                return;
            }

            try {
                m_train_source =
                    std::filesystem::path{ json.at( "train_source" ) };
                if ( !std::filesystem::directory_entry( m_train_source )
                          .exists() ) {
                    m_successful_load = false;
                    return;
                }
            }
            catch ( const nlohmann::json::parse_error & e ) {
                std::cerr << std::format(
                    "No train_source listed in test control config: {}\n{}",
                    test_control_config_path.string(), e.what() )
                          << std::endl;
                m_successful_load = false;
                return;
            }

            try {
                m_test_source =
                    std::filesystem::path{ json.at( "test_source" ) };
                if ( !std::filesystem::directory_entry( m_test_source )
                          .exists() ) {
                    m_successful_load = false;
                    return;
                }
            }
            catch ( const nlohmann::json::parse_error & e ) {
                std::cerr << std::format(
                    "No test_source listed in test control config: {}\n{}",
                    test_control_config_path.string(), e.what() )
                          << std::endl;
                m_successful_load = false;
                return;
            }

            m_forecast_output =
                std::filesystem::path{ json.value( "forecast_output", "" ) };
            m_metadata_output =
                std::filesystem::path{ json.value( "metadata_output", "" ) };

            if ( m_forecast_output == std::string{ "" }
                 && m_metadata_output == std::string{ "" } ) {
                std::cerr << std::format(
                    "No forecast_output or metadata_output listed in test "
                    "control config: {}",
                    test_control_config_path.string() )
                          << std::endl;
                m_successful_load = false;
                return;
            }
            else if ( m_forecast_output == "" ) {
                m_forecast_output = m_metadata_output;
            }
            else if ( m_metadata_output == "" ) {
                m_metadata_output = m_forecast_output;
            }

            if ( !std::filesystem::directory_entry( m_forecast_output ).exists()
                 || !std::filesystem::directory_entry( m_metadata_output )
                         .exists() ) {
                std::cerr << std::format(
                    "forecast_output & metadata_ouput listed in {} must exist.",
                    test_control_config_path.string() )
                          << std::endl;
                m_successful_load = false;
                return;
            }

            try {
                const auto test_files =
                    json["test_files"].get<std::vector<std::string>>();

                for ( const auto & file : test_files ) {
                    const std::filesystem::directory_entry filepath{
                        m_config_source /= file
                    };
                    m_test_files[file] = filepath.exists();
                }
            }
            catch ( const nlohmann::json::parse_error & e ) {
                std::cerr << std::format(
                    "test_files key must be specified in test control config "
                    "at {}",
                    test_control_config_path.string() )
                          << std::endl;
                m_successful_load = false;
                return;
            }
        }
        else {
            m_successful_load = false;
            return;
        }
    }
};

struct TestConfig
{
    std::vector<std::filesystem::path> m_train_test_files;
    nlohmann::json                     m_test_params;

    [[nodiscard]] constexpr inline auto & train_files() const noexcept {
        return m_train_test_files;
    }
    [[nodiscard]] inline auto & test_params() const noexcept {
        return m_test_params;
    }

    TestConfig( const std::filesystem::path & filepath ) {
        std::ifstream fp( filepath, std::ios_base::binary );
        auto          json = nlohmann::json::parse( fp );
        try {
            m_train_test_files = json.at( "train_test_files" );
        }
        catch ( const nlohmann::json::parse_error & e ) {
            std::cerr << std::format(
                "Test config must specify the train_test_files key. It can "
                "either "
                "be empty, [], or specify pairs training and testing data "
                "files.\n" );
        }

        try {
            m_test_params = json.at( "test_params" );
        }
        catch ( const nlohmann::json::parse_error & e ) {
            std::cerr << std::format(
                "Test config must specify the test params in the "
                "\"test_params\" key. Parameters are specified as key: array "
                "pairs.\n" );
        }
    }
};

template <Weight T>
struct ResultMetadata
{
    std::filesystem::path m_config_loc;
    bool                  m_load_success;

    Index  m_n_total_tests;
    Index  m_n_failed_tests;
    double m_test_time;
};

template <Weight T>
class HyperOpt
{
    private:
    TestControlConfig       m_control_config;
    std::vector<TestConfig> m_test_configs;

    public:
    HyperOpt( const std::filesystem::path & control_config_path ) :
        m_control_config( control_config_path ) {
        if ( m_control_config.success() ) {
            for ( const auto & [filename, load_status] :
                  m_control_config.test_files() ) {}
        }
        else {}
    }
};

}; // namespace NVAR
