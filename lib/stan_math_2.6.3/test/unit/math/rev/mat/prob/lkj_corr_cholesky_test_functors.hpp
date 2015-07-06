#include <stan/math/rev/mat/functor/gradient.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_cholesky_log.hpp>
#include <stan/math/prim/mat/functor/finite_diff_gradient.hpp>
#include <stan/math/prim/mat/fun/cholesky_corr_constrain.hpp>
#include <stan/math/prim/scal/fun/positive_constrain.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <vector>

namespace stan {
  
  namespace math {

    template <typename T_L, typename T_eta>
    typename return_type<T_eta, T_L>::type
    lkj_corr_cholesky_uc(Eigen::Matrix<T_L, Eigen::Dynamic, 1> L, T_eta eta, int K) {
      using math::lkj_corr_cholesky_log;
      using math::cholesky_corr_constrain;
      using math::positive_constrain;

      typedef typename return_type<T_eta, T_L>::type rettype;
      rettype lp(0.0);
      Eigen::Matrix<T_L, Eigen::Dynamic, Eigen::Dynamic> L_c
        = cholesky_corr_constrain(L, K, lp);
      T_eta eta_c = positive_constrain(eta, lp);
      lp += lkj_corr_cholesky_log(L_c, eta_c);
      return lp;
    }

    struct lkj_corr_cholesky_cd {
      int K;
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> L_def;
      lkj_corr_cholesky_cd(int K_) : K(K_) {
        using math::cholesky_corr_constrain;
        int k_choose_2 = K * (K - 1) / 2;
        Eigen::Matrix<double, Eigen::Dynamic, 1> L_uc(k_choose_2);
        for (int i = 0; i < k_choose_2; ++i)
          L_uc(i) = i / 10.0;
        L_def = cholesky_corr_constrain(L_uc, K);
      }
      template <typename T>
      T operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> vec) const {
        using math::positive_constrain;
        using math::lkj_corr_cholesky_log;
        T lp(0.0);
        T eta_c = positive_constrain(vec(0), lp);
        lp += lkj_corr_cholesky_log(L_def, eta_c);
        return lp;
      }
    };

    struct lkj_corr_cholesky_dc {
      int K;
      double eta_c;
      lkj_corr_cholesky_dc(int K_) : K(K_) {
        using math::positive_constrain;
        eta_c = positive_constrain(0.5);
      }
      template <typename T>
      T operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> vec) const {
        using math::lkj_corr_cholesky_log;
        using math::cholesky_corr_constrain;
        T lp(0.0);
        Eigen::Matrix<T, -1, -1> L_c = cholesky_corr_constrain(vec, K, lp);
        lp += lkj_corr_cholesky_log(L_c, eta_c);
        return lp;
      }
    };

    struct lkj_corr_cholesky_dd {
      int K;
      lkj_corr_cholesky_dd(int K_) : K(K_) { }
      template <typename T>
      T operator()(Eigen::Matrix<T, Eigen::Dynamic, 1> vec) const {
        int k_choose_2 = K * (K - 1) / 2;
        Eigen::Matrix<T, Eigen::Dynamic, 1> L
          = vec.tail(k_choose_2);
        return lkj_corr_cholesky_uc(L, vec(0), K);
      }
    };
  }

}
