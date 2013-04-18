#ifndef __STAN__MATH__REP_ARRAY_HPP__
#define __STAN__MATH__REP_ARRAY_HPP__

#include <stdexcept>
#include <vector>

#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace math {

    inline void validate_non_negative_rep(int n, const std::string& fun) {
      if (n >= 0) return;
      std::stringstream msg;
      msg << "error in " << fun
          << "; number of replications must be positive"
          << "; found n=" << n;
      throw std::domain_error(msg.str());
    }

    template <typename T>
    inline std::vector<T>
    rep_array(const T& x, int n) {
      validate_non_negative_rep(n,"rep_array 1D");
      return std::vector<T>(n,x);
    }

    template <typename T>
    inline std::vector<std::vector<T> >
    rep_array(const T& x, int m, int n) {
      using std::vector;
      validate_non_negative_rep(m,"rep_array 2D rows");
      validate_non_negative_rep(n,"rep_array 2D cols");
      return vector<vector<T> >(m, vector<T>(n, x));
    }

    template <typename T>
    inline std::vector<std::vector<std::vector<T> > >
    rep_array(const T& x, int k, int m, int n) {
      using std::vector;
      validate_non_negative_rep(k,"rep_array 2D shelfs");
      validate_non_negative_rep(m,"rep_array 2D rows");
      validate_non_negative_rep(n,"rep_array 2D cols");
      return vector<vector<vector<T> > >(k,
                                         vector<vector<T> >(m,
                                                            vector<T>(n, x)));
    }

  }
}

#endif
