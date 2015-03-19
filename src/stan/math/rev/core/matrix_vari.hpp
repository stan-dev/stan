#ifndef STAN__MATH__REV__CORE__MATRIX_VARI_HPP
#define STAN__MATH__REV__CORE__MATRIX_VARI_HPP

#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/core/vari.hpp>

namespace stan {
  namespace agrad {

    class op_matrix_vari : public vari {
    protected:
      const size_t size_;
      vari** vis_;
    public:
      template <int R, int C>
      op_matrix_vari(double f, const Eigen::Matrix<stan::agrad::var,R,C>& vs) :
        vari(f),
        size_(vs.size()) {

        vis_ = (vari**) operator new(sizeof(vari*) * vs.size());
        for (int i = 0; i < vs.size(); ++i)
          vis_[i] = vs(i).vi_;
      }
      vari* operator[](size_t n) const {
        return vis_[n];
      }
      size_t size() {
        return size_;
      }
    };

  }
}
#endif
