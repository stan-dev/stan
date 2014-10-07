#ifndef STAN__MATH__INDEXING__INDEX_HPP
#define STAN__MATH__INDEXING__INDEX_HPP

#include <vector>
#include <stan/meta/indexed_type.hpp>

namespace stan {
  
  namespace math {


    // SINGLE INDEXING (reduces dimensionality)

    /**
     * Structure for an indexing consisting of a single index.
     * Applying this index reduces the dimensionality of the container
     * to which it is applied by one.
     */
    struct index_uni {
      
      /**
       * Type indicating that index_uni is a single index that reduces
       * dimensionality when applied.
       */
      typedef meta::uni_index index_type;

      int n_;

      /**
       * Construct a single indexing from the specified index.
       *
       * @param n single index.
       */
      index_uni(int n) 
      : n_(n) {
      }

    };



    // MULTIPLE INDEXING (does not reduce dimensionality)


    /**
     * Structure for an indexing consisting of multiple indexes.  The
     * indexes do not need to be unique or in order.
     */
    struct index_multi {

      /*
       * Type indicating that index_multi is a multiple indexing that
       * does not reduce dimensionality when applied.
       */
      typedef meta::multi_index index_type;

      std::vector<int> ns_;

      /**
       * Construct a multiple indexing from the specified indexes.
       *
       * @param ns multiple indexes.
       */
      index_multi(const std::vector<int>& ns) 
        : ns_(ns) { 
      }

    };


    /**
     * Structure for an indexing that consists of all indexes for a
     * container.  Applying this index is a no-op.
     */
    struct index_omni {

      /*
       * Type indicating that index_omni is a multiple indexing that
       * does not reduce dimensionality when applied.
       */
      typedef meta::multi_index index_type;
    };


    /**
     * Structure for an indexing from a minimum index (inclusive) to
     * the end of a container. 
     */
    struct index_min {
      
      /**
       * Type indicating that index_min is a multiple indexing that
       * does not reduce dimensionality when applied.
       */
      typedef meta::multi_index index_type;
      
      int min_;

      /**
       * Construct an indexing from the specified minimum index (inclusive).
       *
       * @param min minimum index (inclusive).
       */
      index_min(int min) 
      : min_(min) {
      }

    };


    /**
     * Structure for an indexing from the start of a container to a
     * specified maximum index (inclusive).
     */
    struct index_max {

      /**
       * Type indicating that index_max is a multiple indexing that
       * does not reduce dimensionality when applied.
       */
      typedef meta::multi_index index_type;
      
      int max_;

      /**
       * Construct an indexing from the start of the container up to
       * the specified maximum index (inclusive).
       *
       * @param max maximum index (inclusive).
       */
      index_max(int max) 
      : max_(max) {
      }

    };



    /**
     * Structure for an indexing from a minimum index (inclusive) to a
     * maximum index (inclusive).
     */
    struct index_min_max {

      /**
       * Type indicating that index_min_max is a multiple indexing that
       * does not reduce dimensionality when applied.
       */
      typedef meta::multi_index index_type;

      int min_;
      int max_;

      /**
       * Construct an indexing from the specified minimum index
       * (inclusive) and maximum index (inclusive). 
       *
       * @param min minimum index (inclusive).
       * @param max maximum index (inclusive).
       */
      index_min_max(int min, int max) 
        : min_(min), max_(max) {
      }

    };


  }

}

#endif
