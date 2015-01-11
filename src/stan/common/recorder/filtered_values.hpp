#ifndef STAN__COMMON__RECORDER__FILTERED_VALUES_HPP
#define STAN__COMMON__RECORDER__FILTERED_VALUES_HPP

#include <stan/common/recorder/values.hpp>
#include <ostream>
#include <string>
#include <stdexcept>
#include <vector>

namespace stan {
  namespace common {
    namespace recorder {
      
      template <class InternalVector>
      class filtered_values {
      private:
        size_t N_, M_, N_filter_;
        std::vector<size_t> filter_;
        values<InternalVector> values_;
        std::vector<double> tmp;
        
      public:

        filtered_values(const size_t N,
                        const size_t M,
                        const std::vector<size_t>& filter)
          : N_(N), M_(M), N_filter_(filter.size()), filter_(filter), 
            values_(N_filter_, M_), tmp(N_filter_) { 
          for (size_t n = 0; n < N_filter_; n++)
            if (filter.at(n) >= N_)
              throw std::out_of_range("filter is looking for elements out of range");
        }
        
        filtered_values(const size_t N,
                        const std::vector<InternalVector>& x,
                        const std::vector<size_t>& filter)
          : N_(N), M_(0), filter_(filter), N_filter_(filter.size()),
            values_(x), tmp(N_filter_) {
          if (x.size() != filter.size())
            throw std::length_error("filter provided does not match dimensions of the storage");
          if (N_filter_ > 0)
            M_ = x[0].size();
          for (size_t n = 0; n < N_filter_; n++)
            if (filter.at(n) >= N_)
              throw std::out_of_range("filter is looking for elements out of range");
        }

        void operator()(const std::vector<std::string>& x) { 
          values_(x);
        }

        template <class T>
        void operator()(const std::vector<T>& x) {
          if (x.size() != N_)
            throw std::length_error("vector provided does not match the parameter length");
          for (size_t n = 0; n < N_filter_; n++)
            tmp[n] = x[filter_[n]];
          values_(tmp);
        }
      
        void operator()(const std::string x) {
          values_(x);
        }
      
        void operator()() { 
          values_();
        }
      
        bool is_recording() const {
          return values_.is_recording();
        }

        const std::vector<InternalVector>& x() {
          return values_.x();
        }
      };

    }
  }
}

#endif
