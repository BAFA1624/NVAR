#pragma once

#include "NVAR_util.hpp"

namespace NVAR
{

template <Weight T, bool Transparent = true, Index d = -1, Index n = -1>
class Pipe
{
    private:
    Index                                   m_dim;
    Index                                   m_n_passthrough;
    Vec<T, ( d < 0 || n < 0 ) ? -1 : n>     m_pass_indices;
    Vec<T, ( d < 0 || n < 0 ) ? -1 : d - n> m_divert_indices;

    public:
    constexpr Pipe( const RefIndices<n> pass_indices ) :
        m_dim( d ), m_n_passthrough( n ), m_pass_indices( pass_indices ) {
        static_assert( Transparent || n < d, "Either set transparent " );

        if constexpr ( !Transparent ) {
            Index count{ 0 };
            for ( Index i{ 0 }; i < d; ++i ) {
                bool passed{ false };
                for ( Index j{ 0 }; j < n; ++j ) {
                    if ( m_pass_indices[j] == i ) {
                        passed = true;
                        break;
                    }
                }
                if ( !passed ) {
                    m_divert_indices[count++] = i;
                }
            }
        }
    }
    Pipe( const RefIndices<-1> pass_indices, const Index dim )
        requires( d == Index{ -1 } && n == Index{ -1 } )
        :
        m_dim( dim ),
        m_n_passthrough( dim - pass_indices.size() ),
        m_pass_indices( pass_indices ) {
        if constexpr ( !Transparent ) {
            m_divert_indices = Vec<T>( m_dim - m_pass_indices.size() );
            Index count{ 0 };
            for ( Index i{ 0 }; i < dim; ++i ) {
                bool passed{ false };
                for ( Index j{ 0 }; j < pass_indices.size(); ++j ) {
                    if ( m_pass_indices[j] == i ) {
                        passed = true;
                        break;
                    }
                }
                if ( !passed ) {
                    m_divert_indices[count++] = i;
                }
            }
        }
    }
    Pipe( const std::vector<Index> & pass_indices, const Index dim )
        requires( d == Index{ -1 } && n == Index{ -1 } )
        :
        m_dim( dim ),
        m_n_passthrough( dim - static_cast<Index>( pass_indices.size() ) ),
        m_pass_indices( pass_indices.data(), pass_indices.size() ) {
        if constexpr ( !Transparent ) {
            m_divert_indices = Vec<T>( m_dim - m_pass_indices.size() );
            Index count{ 0 };
            for ( Index i{ 0 }; i < dim; ++i ) {
                bool passed{ false };
                for ( Index j{ 0 }; j < pass_indices.size(); ++j ) {
                    if ( m_pass_indices[j] == i ) {
                        passed = true;
                        break;
                    }
                }
                if ( !passed ) {
                    m_divert_indices[count++] = i;
                }
            }
        }
    }

    constexpr inline Vec<T, d>
    forward( const ConstRefVec<T, d>                      v,
             [[maybe_unused]] const ConstRefVec<T, d - n> vals ) {
        if constexpr ( Transparent ) {
            return v;
        }
        else {
            Vec<T, d> result;
            for ( Index i{ 0 }; i < n; ++i ) {
                result[m_pass_indices[i]] = v[m_pass_indices[i]];
            }
            for ( Index i{ 0 }; i < d - n; ++i ) {
                result[m_divert_indices[i]] = vals[i];
            }
            return result;
        }
    }

    constexpr inline Vec<T, -1>
    forward( const ConstRefVec<T, -1>                  v,
             [[maybe_unused]] const ConstRefVec<T, -1> vals )
        requires( d == Index{ -1 } && n == Index{ -1 } )
    {
        if constexpr ( Transparent ) {
            return v;
        }
        else {
            Vec<T> result( v.size() );
            for ( Index i{ 0 }; i < m_dim; ++i ) {
                result[m_pass_indices[i]] = v[m_pass_indices[i]];
            }
            for ( Index i{ 0 }; i < m_dim - m_n_passthrough; ++i ) {
                result[m_divert_indices[i]] = vals[i];
            }
            return result;
        }
    }
};

} // namespace NVAR
