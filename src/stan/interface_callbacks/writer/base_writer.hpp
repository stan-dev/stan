#ifndef STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_BASE_WRITER_HPP

#include <boost/lexical_cast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {


      /**
       * base_writer is an abstract base class defining the interface
       * for Stan writer callbacks.
       * 
       * The Stan API writes structured output to implementations of
       * this class defined by a given interface.
       */
      class base_writer {
      public:
        /**
         * Writes a key, value pair.
         *
         * @param[in] key A string
         * @param[in] value A double value
         */
        virtual void operator()(const std::string& key,
                                double value) = 0;

        /**
         * Writes a key, value pair.
         *
         * @param[in] key A string
         * @param[in] value An integer value
         */
        virtual void operator()(const std::string& key,
                                int value) = 0;

        /**
         * Writes a key, value pair.
         *
         * @param[in] key A string
         * @param[in] value A string
         */
        virtual void operator()(const std::string& key,
                                const std::string& value) = 0;

        /**
         * Writes a key, value pair.
         *
         * @param[in] key A string
         * @param[in] values A double array, typically used with
         *   contiguous Eigen vectors
         * @param[in] n_values Length of the array
         */
        virtual void operator()(const std::string& key,
                                const double* values,
                                int n_values) = 0;

        /**
         * Writes a key, value pair.
         *
         * @param[in] key A string
         * @param[in] values A double array assumed to represent a 2d 
         *   matrix stored in column major order, typically used with
         *   contiguous Eigen matrices
         * @param[in] n_rows Rows
         * @param[in] n_cols Columns
         */
        virtual void operator()(const std::string& key,
                                const double* values,
                                int n_rows, int n_cols) = 0;

        /**
         * Writes a set of names.
         *
         * @param[in] names Names in a std::vector
         */
        virtual void operator()(const std::vector<std::string>& names) = 0;

        /**
         * Writes a set of values.
         *
         * @param[in] state Values in a std::vector
         */
        virtual void operator()(const std::vector<double>& state) = 0;

        /**
         * Writes blank input.
         */
        virtual void operator()() = 0;

        /**
         * Writes a string.
         *
         * @param[in] message A string
         */
        virtual void operator()(const std::string& message) = 0;

        /**
         * Destructor.
         *
         * Virtual destructor to avoid compiler warnings
         *
         */
        virtual ~base_writer() {}
      };

    }
  }
}

#endif
