#ifndef STAN_INTERFACE_CALLBACKS_WRITER_FILTERED_VALUES_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_FILTERED_VALUES_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/interface_callbacks/writer/values.hpp>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      /**
       * Writes a subset of values to a pre-allocated vector. Used by
       * interfaces.
       *
       * @tparam InternalVector A type that has the interface of a
       * std::vector
       */
      template <class InternalVector>
      class filtered_values: public base_writer {
      private:
        size_t N_, M_, N_include_;
        std::vector<size_t> include_;
        values<InternalVector> values_;
        std::vector<double> tmp;

      public:
        /**
         * Constructor with given size and a list of elements to include.
         *
         * @param N size of individual vector
         * @param M number of vectors
         * @param include a list of indices (0-indexed) to be included;
         *    the elements not included are not written.
         */
        filtered_values(const size_t N,
                        const size_t M,
                        const std::vector<size_t>& include)
          : N_(N), M_(M), N_include_(include.size()), include_(include),
            values_(N_include_, M_), tmp(N_include_) {
          for (size_t n = 0; n < N_include_; n++)
            if (include.at(n) >= N_)
              throw std::out_of_range("filter is looking for "
                                      "elements out of range");
        }

        /**
         * Constructor that accepts a vector of individual vectors
         * that have bene allocated outside this call and
         * a list of elements to include
         *
         * @param N size of individual vector
         * @param x a vector of individual vectors that have been allocated
         *   from outside. The size of the individual vectors here should be
         *   the length of the include argument.
         * @param include a list of indices (0-indexed) to be included;
         *    the elements not included are not written.
         */
        filtered_values(const size_t N,
                        const std::vector<InternalVector>& x,
                        const std::vector<size_t>& include)
          : N_(N), M_(0), include_(include), N_include_(include.size()),
            values_(x), tmp(N_include_) {
          if (x.size() != include.size())
            throw std::length_error("filter provided does not "
                                    "match dimensions of the storage");
          if (N_include_ > 0)
            M_ = x[0].size();
          for (size_t n = 0; n < N_include_; n++)
            if (include.at(n) >= N_)
              throw std::out_of_range("filter is looking for "
                                      "elements out of range");
        }


                
        /**
         * Writes a key, value pair.
         * This implementation does nothing.
         *
         * @param key The key
         * @param value The value
         */
        void operator()(const std::string& key,
                        double value) { }

        /**
         * Writes a key, value pair.
         * This implementation does nothing.
         *
         * @param key The key
         * @param value The value
         */
        void operator()(const std::string& key,
                        const std::string& value) { }
        
        /**
         * Writes a key and an array of values.
         * This implementation does nothing.
         *
         * @param key The key
         * @param values The values
         * @param n_values The number of values
         */
        void operator()(const std::string& key,
                                const double* values,
                                int n_values)  { }

        /**
         * Writes a key and a 2-d array of values.
         * This implementation does nothing.
         *
         * @param key The key
         * @param values The values stored in row-major order
         * @param n_rows The number of rows
         * @param n_cols The number of columns
         */
        void operator()(const std::string& key,
                                const double* values,
                                int n_rows, int n_cols) { }


        /**
         * Writes a vector of names. This implementation does nothing.
         *
         * @param names Names to write
         */
        void operator()(const std::vector<std::string>& names) { }

        /**
         * Writes a message.
         * This implementation does nothing.
         *
         * @param message Message to write
         */
        void operator()(const std::string& message) { }

        /**
         * Writes nothing.
         * This implementation does nothing.
         */
        void operator()() { }

        /**
         * Writes a vector of states.
         * Only writes a subset of these states, defined at
         * construction.
         *
         * @param state State to write
         */
        void operator()(const std::vector<double>& state) {
          if (state.size() != N_)
            throw std::length_error("vector provided does not "
                                    "match the parameter length");
          for (size_t n = 0; n < N_include_; n++)
            tmp[n] = state[include_[n]];
          values_(tmp);
        }

        /**
         * Accesses the vector of values.
         *
         * @return a const reference to the vector of internal vectors
         */
        const std::vector<InternalVector>& x() {
          return values_.x();
        }
      };

    }
  }
}

#endif
