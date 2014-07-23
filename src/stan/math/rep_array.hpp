#ifndef STAN__MATH__REP_ARRAY_HPP
#define STAN__MATH__REP_ARRAY_HPP

#include <vector>

#include <stan/math/error_handling/check_nonnegative.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline std::vector<T>
    rep_array(const T& x, int n) {
      check_nonnegative("rep_array(%1%)", n,"n", (double*)0);
      return std::vector<T>(n,x);
    }

    template <typename T>
    inline std::vector<std::vector<T> >
    rep_array(const T& x, int m, int n) {
      using std::vector;
      check_nonnegative("rep_array(%1%)", m,"rows", (double*)0);
      check_nonnegative("rep_array(%1%)", n,"cols", (double*)0);
      return vector<vector<T> >(m, vector<T>(n, x));
    }

    template <typename T>
    inline std::vector<std::vector<std::vector<T> > >
    rep_array(const T& x, int k, int m, int n) {
      using std::vector;
      check_nonnegative("rep_array(%1%)", k,"shelfs", (double*)0);
      check_nonnegative("rep_array(%1%)", m,"rows", (double*)0);
      check_nonnegative("rep_array(%1%)", n,"cols", (double*)0);
      return vector<vector<vector<T> > >(k,
                                         vector<vector<T> >(m,
                                                            vector<T>(n, x)));
    }

  }
}

#endif
