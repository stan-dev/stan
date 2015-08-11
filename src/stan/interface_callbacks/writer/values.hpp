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

      template <class InternalVector>
      class values: public base_writer {
      private:
        size_t m_;
        size_t N_;
        size_t M_;
        std::vector<InternalVector> x_;

      public:
        values(const size_t N,
               const size_t M)
          : m_(0), N_(N), M_(M) {
          x_.reserve(N_);
          for (size_t n = 0; n < N_; n++)
            x_.push_back(InternalVector(M_));
        }

        explicit values(const std::vector<InternalVector>& x)
          : m_(0), N_(x.size()), M_(0),
            x_(x) {
          if (N_ > 0)
            M_ = x_[0].size();
        }

        void operator()(const std::vector<std::string>& x) { }

        template <class T>
        void operator()(const std::vector<T>& x) {
          if (N_ != x.size())
            throw std::length_error("vector provided does not "
                                    "match the parameter length");
          if (m_ == M_)
            throw std::out_of_range("");
          for (size_t n = 0; n < N_; n++)
            x_[n][m_] = x[n];
          m_++;
        }

        void operator()(const std::string& message) { }

        void operator()() { }

        bool is_writing() const {
          if (m_ < M_)
            return true;
          return false;
        }

        const std::vector<InternalVector>& x() const {
          return x_;
        }
      };

    }
  }
}

#endif
