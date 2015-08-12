#ifndef STAN_INTERFACE_CALLBACKS_WRITER_VALUES_HPP
#define STAN_INTERFACE_CALLBACKS_WRITER_VALUES_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {
  namespace interface_callbacks {
    namespace writer {

      /**
       * Writes values to a pre-allocated vector. Used by interfaces.
       *
       * @tparam InternalVector A type that has the interface of a
       * std::vector
       */
      template <class InternalVector>
      class values: public base_writer {
      private:
        size_t m_;
        size_t N_;
        size_t M_;
        std::vector<InternalVector> x_;

      public:
        /**
         * Constructor with given size.
         *
         * @param N size of individual vector
         * @param M number of vectors
         */
        values(const size_t N,
               const size_t M)
          : m_(0), N_(N), M_(M) {
          x_.reserve(N_);
          for (size_t n = 0; n < N_; n++)
            x_.push_back(InternalVector(M_));
        }

        /**
         * Constructor that accepts a vector of individual vectors
         * that have been allocated outside this call.
         *
         * Each internal vector must be the same length.
         *
         * @param[in,out] x The externally allocated vector to populate.
         */
        explicit values(const std::vector<InternalVector>& x)
          : m_(0), N_(x.size()), M_(0), x_(x) {
          if (N_ > 0)
            M_ = x_[0].size();
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
         * Writes a vector of states.
         *
         * @param state State to write
         */
        void operator()(const std::vector<double>& state) {
          if (N_ != state.size())
            throw std::length_error("vector provided does not "
                                    "match the parameter length");
          if (m_ == M_)
            throw std::out_of_range("");
          for (size_t n = 0; n < N_; n++)
            x_[n][m_] = state[n];
          m_++;
        }

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
         * Accesses the vector of values.
         *
         * @return a const reference to the vector of internal vectors
         */
        const std::vector<InternalVector>& x() const {
          return x_;
        }
      };

    }
  }
}

#endif
