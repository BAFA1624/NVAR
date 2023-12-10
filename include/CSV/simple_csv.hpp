#pragma once

#include "NVAR/NVAR_util.hpp"

#include <algorithm>
#include <concepts>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace CSV
{

template <typename T>
concept Readable = std::floating_point<T> || std::integral<T>
                   || std::convertible_to<T, std::string>;

class SimpleCSV
{
    private:
    // Success / Fail
    bool m_successful;

    // Data
    std::map<std::string, NVAR::Index> m_title_map;
    std::vector<std::vector<double>>   m_data;
    NVAR::Index                        m_rows;
    NVAR::Index                        m_cols;

    // Parsing information
    std::filesystem::directory_entry m_file;
    bool                             m_col_titles;
    NVAR::Index                      m_skip_header;
    std::string_view                 m_delim;
    NVAR::Index                      m_max_line_size;

    // Parsing
    void parse() noexcept;

    public:
    SimpleCSV( const std::filesystem::path filepath,
               const bool col_titles = false, const NVAR::Index skip_header = 0,
               const std::string_view delim = ",",
               const NVAR::Index      max_line_size = 256 ) :
        m_successful( false ),
        m_file( filepath ),
        m_col_titles( col_titles ),
        m_skip_header( skip_header ),
        m_delim( delim ),
        m_max_line_size( max_line_size ) {
        if ( m_file.exists() ) {
            parse();
            m_successful = true;
        }
        else {
            std::cerr << m_file.path().string() << " does not exist."
                      << std::endl;
            exit( EXIT_FAILURE );
        }
    }

    [[nodiscard]] constexpr inline auto success() const noexcept {
        return m_successful;
    }

    template <typename T = double>
    [[nodiscard]] constexpr std::vector<T>
    col( const NVAR::Index i ) const noexcept;
    template <typename T = double>
    [[nodiscard]] constexpr NVAR::Mat<T, -1, 1>
    colv( const NVAR::Index i ) const noexcept;
    template <typename T = double, NVAR::Index R = -1, NVAR::Index C = -1>
    [[nodiscard]] constexpr NVAR::Mat<T, R, C>
    atv( const NVAR::Index row_offset = 0,
         const NVAR::Index col_offset = 0 ) const noexcept;

    template <typename T = double, NVAR::Index R = -1, NVAR::Index C = -1>
    [[nodiscard]] static inline bool
    write( const std::filesystem::path file, const NVAR::RefMat<T, R, C> m,
           const std::vector<std::string> & titles = {} ) noexcept;
    template <typename T>
    [[nodiscard]] static inline bool
    write( const std::filesystem::path file, const NVAR::RefMat<T, -1, -1> m,
           const std::vector<std::string> & titles = {} ) noexcept;

    [[nodiscard]] constexpr auto & rows() const noexcept { return m_rows; };
    [[nodiscard]] constexpr auto & cols() const noexcept { return m_cols; };
    [[nodiscard]] constexpr std::vector<std::string>
    col_titles() const noexcept {
        std::vector<std::string> v;
        const auto               view{ std::views::keys( m_title_map ) };
        if constexpr ( std::ranges::sized_range<decltype( view )> ) {
            v.reserve( std::ranges::size( view ) );
        }
        std::ranges::copy( view, std::back_inserter( v ) );
        return v;
    }
};

void
SimpleCSV::parse() noexcept {
    std::ifstream file{ m_file.path(),
                        std::ifstream::ate | std::ifstream::binary };

    m_successful = false;
    if ( !file.is_open() ) {
        std::cerr << "Failed to open file." << std::endl;
        exit( EXIT_FAILURE );
    }
    else {
        // Get size of file
        const NVAR::Index full_file_sz{ file.tellg() };
        file.seekg( file.beg );

        // Skip header
        for ( NVAR::Index i{ 0 }; i < m_skip_header; ++i ) {
            file.ignore( m_max_line_size, '\n' );
        }

        const NVAR::Index start_pos{ file.tellg() };
        const NVAR::Index file_sz{ full_file_sz - start_pos };

        // Read all text from file
        std::string file_str( static_cast<std::size_t>( file_sz ), '\0' );
        if ( !file.read( &file_str[0], file_sz ) ) {
            std::cerr << "Failed to read file data." << std::endl;
            exit( EXIT_FAILURE );
        }

        // Add to stringstream
        std::stringstream ss( file_str );
        std::string       line;

        // Parse col title if there are any
        if ( m_col_titles ) {
            if ( std::getline( ss, line, '\n' ) ) {
                auto split_row{ std::views::split( line, m_delim )
                                | std::views::enumerate };
                for ( const auto & [i, title] : split_row ) {
                    m_title_map[std::string{ title.data(), title.size() }] = i;
                }
            }
            else {
                std::cerr << "Failed to parse column titles." << std::endl;
                exit( EXIT_FAILURE );
            }
        }
        else {
            // If no titles, get no. cols in first row
            if ( std::getline( ss, line, '\n' ) ) {
                auto split_row{ std::views::split( line, m_delim )
                                | std::views::enumerate };
                for ( const auto & [i, val] : split_row ) {
                    m_title_map[std::to_string( i )] = i;
                }
                ss = std::stringstream( file_str );
            }
            else {
                std::cerr << "Unable to count number of columns." << std::endl;
                exit( EXIT_FAILURE );
            }
        }

        // Set size of m_data
        m_cols = static_cast<NVAR::Index>( m_title_map.size() );
        m_data = std::vector<std::vector<double>>( m_title_map.size() );

        // Parse file line-by-line
        NVAR::Index row_count{ 0 };
        while ( std::getline( ss, line, '\n' ) ) {
            auto split_row{ std::views::split( std::string_view( line ),
                                               m_delim )
                            | std::views::enumerate };

            // Read values in
            NVAR::Index count{ 0 }; // Count to ensure no. cols is consistent
            for ( const auto & [i, value] : split_row ) {
                m_data[static_cast<std::size_t>( i )].push_back(
                    std::stod( std::string{ value.data(), value.size() } ) );
                count++;
            }

            // Check no. of vals on row matches expected no.
            if ( count != static_cast<NVAR::Index>( m_title_map.size() ) ) {
                m_data.clear();
                m_title_map.clear();
                std::cerr << "Inconsistent number of columns detected."
                          << std::endl;
                exit( EXIT_FAILURE );
            }

            row_count++;
        }

        m_rows = row_count;
        m_successful = true;
        return;
    }
}

template <typename T>
constexpr std::vector<T>
SimpleCSV::col( const NVAR::Index i ) const noexcept {
    if ( m_successful ) {
        return m_data[static_cast<std::size_t>( i )];
    }
    else {
        return std::vector<T>{};
    }
}
template <typename T>
constexpr NVAR::Mat<T, -1, 1>
SimpleCSV::colv( const NVAR::Index i ) const noexcept {
    if ( m_successful ) {
        NVAR::Mat<T, -1, 1> result(
            m_data[static_cast<std::size_t>( i )].size() );
        for ( const auto & [j, x] : m_data[i] | std::views::enumerate ) {
            result[j] = x;
        }
        return result;
    }
    else {
        return NVAR::Mat<T, -1, 1>{};
    }
}

template <typename T, NVAR::Index R, NVAR::Index C>
[[nodiscard]] constexpr NVAR::Mat<T, R, C>
SimpleCSV::atv( const NVAR::Index row_offset,
                const NVAR::Index col_offset ) const noexcept {
    if ( m_successful ) {
        if constexpr ( R > 0 && C > 0 ) {
            NVAR::Mat<T, R, C> result;
            for ( NVAR::Index col{ 0 }; col < C; ++col ) {
                for ( NVAR::Index row{ 0 }; row < R; ++row ) {
                    result( row, col ) =
                        m_data[static_cast<std::size_t>( col_offset + col )]
                              [static_cast<std::size_t>( row_offset + row )];
                }
            }
            return result;
        }
        else {
            NVAR::Mat<T, -1, -1> result( m_rows, m_cols );
            for ( NVAR::Index col{ 0 }; col < m_cols; ++col ) {
                for ( NVAR::Index row{ 0 }; row < m_rows; ++row ) {
                    result( row, col ) =
                        m_data[static_cast<std::size_t>( col_offset + col )]
                              [static_cast<std::size_t>( row_offset + row )];
                }
            }
            return result;
        }
    }
    else {
        return NVAR::Mat<T, R, C>{};
    }
}

template <typename T, NVAR::Index R, NVAR::Index C>
inline bool
SimpleCSV::write( const std::filesystem::path      file,
                  const NVAR::RefMat<T, R, C>      m,
                  const std::vector<std::string> & titles ) noexcept {
    std::ofstream fp( file, std::ofstream::out | std::ofstream::binary );
    if ( fp.is_open() ) {
        if ( !titles.empty()
             && static_cast<NVAR::Index>( titles.size() ) == R ) {
            // std::string line;
            for ( const auto c : titles | std::views::join_with( ',' ) ) {
                fp << c;
            }
            // for ( const auto & title : titles ) { line += title + ","; }
            // line.pop_back();
            fp << std::endl;
        }

        for ( NVAR::Index row{ 0 }; row < R; ++row ) {
            std::string line;
            for ( NVAR::Index col{ 0 }; col < C; ++col ) {
                line += std::to_string( m( row, col ) ) + ",";
            }
            line.pop_back();
            fp << line << std::endl;
        }

        return true;
    }

    std::cerr << std::format( "Unable to write to file: {}\n", file.string() );

    return false;
}

template <typename T>
inline bool
SimpleCSV::write( const std::filesystem::path      file,
                  const NVAR::RefMat<T, -1, -1>    m,
                  const std::vector<std::string> & titles ) noexcept {
    std::ofstream fp( file, std::ofstream::out | std::ofstream::binary );
    if ( fp.is_open() ) {
        if ( !titles.empty() ) {
            for ( const auto c : titles | std::views::join_with( ',' ) ) {
                fp << c;
            }
            fp << std::endl;
        }

        for ( NVAR::Index row{ 0 }; row < m.rows(); ++row ) {
            std::string line;
            for ( NVAR::Index col{ 0 }; col < m.cols(); ++col ) {
                line += std::to_string( m( row, col ) ) + ",";
            }
            line.pop_back();
            fp << line << std::endl;
        }

        return true;
    }

    std::cerr << std::format( "Unable to write to file: {}.\n", file.string() );

    return false;
}

}; // namespace CSV
