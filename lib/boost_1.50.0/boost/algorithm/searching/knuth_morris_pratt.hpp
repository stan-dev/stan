/* 
   Copyright (c) Marshall Clow 2010-2012.

   Distributed under the Boost Software License, Version 1.0. (See accompanying
   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

    For more information, see http://www.boost.org
*/

#ifndef BOOST_ALGORITHM_KNUTH_MORRIS_PRATT_SEARCH_HPP
#define BOOST_ALGORITHM_KNUTH_MORRIS_PRATT_SEARCH_HPP

#include <vector>
#include <iterator>     // for std::iterator_traits

#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/algorithm/searching/detail/debugging.hpp>

// #define  BOOST_ALGORITHM_KNUTH_MORRIS_PRATT_DEBUG

namespace boost { namespace algorithm {

// #define  NEW_KMP

/*
    A templated version of the Knuth-Morris-Pratt searching algorithm.
    
    Requirements:
        * Random-access iterators
        * The two iterator types (I1 and I2) must "point to" the same underlying type.

    http://en.wikipedia.org/wiki/Knuth-Morris-Pratt_algorithm
    http://www.inf.fh-flensburg.de/lang/algorithmen/pattern/kmpen.htm
*/

    template <typename patIter>
    class knuth_morris_pratt {
        typedef typename std::iterator_traits<patIter>::difference_type difference_type;
    public:
        knuth_morris_pratt ( patIter first, patIter last ) 
                : pat_first ( first ), pat_last ( last ), 
                  k_pattern_length ( std::distance ( pat_first, pat_last )),
                  skip_ ( k_pattern_length + 1 ) {
#ifdef NEW_KMP
            preKmp ( pat_first, pat_last );
#else
            init_skip_table ( pat_first, pat_last );
#endif
#ifdef BOOST_ALGORITHM_KNUTH_MORRIS_PRATT_DEBUG
            detail::PrintTable ( skip_.begin (), skip_.end ());
#endif
            }
            
        ~knuth_morris_pratt () {}
        
        /// \fn operator ( corpusIter corpus_first, corpusIter corpus_last, Pred p )
        /// \brief Searches the corpus for the pattern that was passed into the constructor
        /// 
        /// \param corpus_first The start of the data to search (Random Access Iterator)
        /// \param corpus_last  One past the end of the data to search
        /// \param p            A predicate used for the search comparisons.
        ///
        template <typename corpusIter>
        corpusIter operator () ( corpusIter corpus_first, corpusIter corpus_last ) const {
            BOOST_STATIC_ASSERT (( boost::is_same<
                typename std::iterator_traits<patIter>::value_type, 
                typename std::iterator_traits<corpusIter>::value_type>::value ));
            if ( corpus_first == corpus_last ) return corpus_last;  // if nothing to search, we didn't find it!
            if ( pat_first == pat_last )       return corpus_first; // empty pattern matches at start

            const difference_type k_corpus_length = std::distance ( corpus_first, corpus_last );
        //  If the pattern is larger than the corpus, we can't find it!
            if ( k_corpus_length < k_pattern_length ) 
                return corpus_last;

            return do_search   ( corpus_first, corpus_last, k_corpus_length );
            }
    
    private:
/// \cond DOXYGEN_HIDE
        patIter pat_first, pat_last;
        const difference_type k_pattern_length;
        std::vector <difference_type> skip_;

        /// \fn operator ( corpusIter corpus_first, corpusIter corpus_last, Pred p )
        /// \brief Searches the corpus for the pattern that was passed into the constructor
        /// 
        /// \param corpus_first The start of the data to search (Random Access Iterator)
        /// \param corpus_last  One past the end of the data to search
        /// \param p            A predicate used for the search comparisons.
        ///
        template <typename corpusIter>
        corpusIter do_search ( corpusIter corpus_first, corpusIter corpus_last, 
                                                difference_type k_corpus_length ) const {
            difference_type match_start = 0;  // position in the corpus that we're matching
            
#ifdef NEW_KMP
            int patternIdx = 0;
            while ( match_start < k_corpus_length ) {
                while ( patternIdx > -1 && pat_first[patternIdx] != corpus_first [match_start] )
                    patternIdx = skip_ [patternIdx]; //<--- Shifting the pattern on mismatch

                patternIdx++;
                match_start++; //<--- corpus is always increased by 1

                if ( patternIdx >= (int) k_pattern_length )
                    return corpus_first + match_start - patternIdx;
                }
            
#else
//  At this point, we know:
//          k_pattern_length <= k_corpus_length
//          for all elements of skip, it holds -1 .. k_pattern_length
//      
//          In the loop, we have the following invariants
//              idx is in the range 0 .. k_pattern_length
//              match_start is in the range 0 .. k_corpus_length - k_pattern_length + 1

            const difference_type last_match = k_corpus_length - k_pattern_length;
            difference_type idx = 0;          // position in the pattern we're comparing

            while ( match_start <= last_match ) {
                while ( pat_first [ idx ] == corpus_first [ match_start + idx ] ) {
                    if ( ++idx == k_pattern_length )
                        return corpus_first + match_start;
                    }
            //  Figure out where to start searching again
           //   assert ( idx - skip_ [ idx ] > 0 ); // we're always moving forward
                match_start += idx - skip_ [ idx ];
                idx = skip_ [ idx ] >= 0 ? skip_ [ idx ] : 0;
           //   assert ( idx >= 0 && idx < k_pattern_length );
                }
#endif
                
        //  We didn't find anything
            return corpus_last;
            }
    

        void preKmp ( patIter first, patIter last ) {
           const /*std::size_t*/ int count = std::distance ( first, last );
        
           int i, j;
        
           i = 0;
           j = skip_[0] = -1;
           while (i < count) {
              while (j > -1 && first[i] != first[j])
                 j = skip_[j];
              i++;
              j++;
              if (first[i] == first[j])
                 skip_[i] = skip_[j];
              else
                 skip_[i] = j;
           }
        }


        void init_skip_table ( patIter first, patIter last ) {
            const difference_type count = std::distance ( first, last );
    
            int j;
            skip_ [ 0 ] = -1;
            for ( int i = 1; i <= count; ++i ) {
                j = skip_ [ i - 1 ];
                while ( j >= 0 ) {
                    if ( first [ j ] == first [ i - 1 ] )
                        break;
                    j = skip_ [ j ];
                    }
                skip_ [ i ] = j + 1;
                }
            }
// \endcond
        };


/// \fn knuth_morris_pratt_search ( corpusIter corpus_first, corpusIter corpus_last, 
///       patIter pat_first, patIter pat_last )
/// \brief Searches the corpus for the pattern.
/// 
/// \param corpus_first The start of the data to search (Random Access Iterator)
/// \param corpus_last  One past the end of the data to search
/// \param pat_first    The start of the pattern to search for (Random Access Iterator)
/// \param pat_last     One past the end of the data to search for
///
    template <typename patIter, typename corpusIter>
    corpusIter knuth_morris_pratt_search ( 
            corpusIter corpus_first, corpusIter corpus_last, 
            patIter pat_first, patIter pat_last ) {
        knuth_morris_pratt<patIter> kmp ( pat_first, pat_last );
        return kmp ( corpus_first, corpus_last );
        }
}}

#endif  // BOOST_ALGORITHM_KNUTH_MORRIS_PRATT_SEARCH_HPP
