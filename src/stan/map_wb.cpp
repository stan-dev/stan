#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <boost/type_traits/conditional.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/scalar_type.hpp>
#include <stan/math/rev/core.hpp>

#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/meta/is_fvar.hpp>
#include <stan/math/fwd/scal/meta/partials_type.hpp>
#include <stan/math/prim/scal/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/contains_fvar.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/partials_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/meta/is_var.hpp>
#include <stan/math/rev/scal/meta/partials_type.hpp>

struct dummy { };

template <typename T, int R, int C>
size_t length(const std::vector<Eigen::Matrix<T, R, C> >& x) {
  size_t len = 0; 
  for (size_t i = 0; i < x.size(); ++i)
    len += x[i].rows() * x[i].cols();
  return len;
}

template <typename T1, typename T2>
class vector_view_map {
  public:
    vector_view_map(const T1& x, T2* y) : y_(y) { }

    T2& operator[](int i) {
      return y_[0];
    }
  private:
    T2* y_;
};

template <typename T2>
class vector_view_map<dummy, T2> {
  public:
    typedef typename stan::scalar_type<T2>::type scalar_t;
    template <typename T1>
    vector_view_map(const T1& x, scalar_t* y){ };
    scalar_t operator[](int n) const {
      throw std::out_of_range("can't access dummy elements.");
    }
};

template <typename T1, typename T2>
class vector_view_map<Eigen::Matrix<T1, -1, 1>, Eigen::Matrix<T2, -1, 1> > {
  public:
    vector_view_map(const Eigen::Matrix<T1, -1, 1>& x, T2* y) 
     : y_(y, x.rows(), x.cols()) { }

    Eigen::Map<Eigen::Matrix<T2, -1, 1> >& operator[](int i) {
      return y_;
    }
  private:
    Eigen::Map<Eigen::Matrix<T2, -1, 1> > y_;
};

template <typename T1, typename T2>
class vector_view_map<Eigen::Matrix<T1, -1, 1>, T2> {
  public:
    vector_view_map(const Eigen::Matrix<T1, -1, 1>& x, T2* y) 
     : y_(y) { }

    T2& operator[](int i) {
      return y_[i];
    }
  private:
    T2* y_;
};

template <typename T1, typename T2>
class vector_view_map<Eigen::Matrix<T1, 1, -1>, Eigen::Matrix<T2, 1, -1> > {
  public:
    vector_view_map(const Eigen::Matrix<T1, 1, -1>& x, T2* y) 
     : y_(y, x.rows(), x.cols()) { }

    Eigen::Map<Eigen::Matrix<T2, 1, -1> >& operator[](int i) {
      return y_;
    }
  private:
    Eigen::Map<Eigen::Matrix<T2, 1, -1> > y_;
};

template <typename T1, typename T2>
class vector_view_map<Eigen::Matrix<T1, 1, -1>, T2> {
  public:
    vector_view_map(const Eigen::Matrix<T1, 1, -1>& x, T2* y) 
     : y_(y) { }

    T2& operator[](int i) {
      return y_[i];
    }
  private:
    T2* y_;
};

template <typename T1, typename T2, int M, int N>
class vector_view_map<Eigen::Matrix<T1, M, N>, Eigen::Matrix<T2, M, N> > {
  public:
    vector_view_map(const Eigen::Matrix<T1, M, N>& x, T2* y) 
     : y_(y, x.rows(), x.cols()) { }

    Eigen::Map<Eigen::Matrix<T2, M, N> >& operator[](int i) {
      return y_;
    }
  private:
    Eigen::Map<Eigen::Matrix<T2, M, N> > y_;
};

template <typename T1, typename T2, int M, int N>
class vector_view_map<std::vector<Eigen::Matrix<T1, M, N> >,
                      Eigen::Matrix<T2, M, N> > {
  public:
    vector_view_map(const std::vector<Eigen::Matrix<T1, M, N> >& x, T2* y) 
     : y_(y, 1, 1), arrstart(y) {
       if(x.size() > 0) {
         rows = x[0].rows();
         cols = x[0].cols();
       }
     }

    Eigen::Map<Eigen::Matrix<T2, M, N> >& operator[](int i) {
      int offset = i * rows * cols; 
      new (&y_) Eigen::Map<Eigen::Matrix<T2, M, N> >(arrstart + offset, rows, cols); 
      return y_;
    }
  private:
    Eigen::Map<Eigen::Matrix<T2, M, N> > y_;
    T2* arrstart;
    int rows;
    int cols;
};

template <typename T1, typename T2>
class vector_view_map<std::vector<T1>, T2> {
  public:
    vector_view_map(const std::vector<T1>& x, T2* y) 
     : y_(y) { }

    T2& operator[](int i) {
      return y_[i];
    }
  private:
    T2* y_;
};



namespace stan {
  namespace agrad {

    class partials_vari : public vari {
    private:
      const size_t N_;
      vari** operands_;
      double* partials_;
    public:
      partials_vari(double value,
                    size_t N,
                    vari** operands, double* partials)
        : vari(value),
          N_(N),
          operands_(operands),
          partials_(partials) { }
      void chain() {
        for (size_t n = 0; n < N_; ++n)
          operands_[n]->adj_ += adj_ * partials_[n];
      }
    };

      template<typename T,
               bool is_vec = is_vector<T>::value,
               bool is_const = is_constant_struct<T>::value,
               bool contain_fvar = contains_fvar<T>::value>
      struct set_varis {
        inline size_t set(agrad::vari** /*varis*/, const T& /*x*/) {
          return 0U;
        }
      };
      template<typename T>
      struct set_varis<T, true, false, false> {
        inline size_t set(agrad::vari** varis, const T& x) {
          for (size_t n = 0; n < length(x); n++)
            varis[n] = x[n].vi_;
          return length(x);
        }
      };
      template<typename T>
      struct set_varis<T, true, false, true> {
        inline size_t set(agrad::vari** varis, const T& x) {
          for (size_t n = 0; n < length(x); n++)
            varis[n] = 0;
          return length(x);
        }
      };
      template<>
      struct set_varis<agrad::var, false, false, false> {
        inline size_t set(agrad::vari** varis, const agrad::var& x) {
          varis[0] = x.vi_;
          return 1;
        }
      };
    

    /**
     * A variable implementation that stores operands and
     * derivatives with respect to the variable.
     */
    template<typename T1 = double, typename C1 = double, 
             typename T2 = double, typename C2 = double,
             typename T3 = double, typename C3 = double,
             typename T4 = double, typename C4 = double,
             typename T5 = double, typename C5 = double,
             typename T6 = double, typename C6 = double>
    struct OperandsAndPartials {
      typedef
      typename stan::partials_return_type<T1, T2, T3, T4, T5, T6>::type
      T_partials_return;

      typedef
      typename stan::return_type<T1, T2, T3, T4, T5, T6>::type T_return_type;

      static const bool all_constant = is_constant<T_return_type>::value;
      typedef typename boost::conditional<stan::is_constant_struct<T1>::value, dummy, T1>::type T1_proc;
      typedef typename boost::conditional<stan::is_constant_struct<T2>::value, dummy, T2>::type T2_proc;
      typedef typename boost::conditional<stan::is_constant_struct<T3>::value, dummy, T3>::type T3_proc;
      typedef typename boost::conditional<stan::is_constant_struct<T4>::value, dummy, T4>::type T4_proc;
      typedef typename boost::conditional<stan::is_constant_struct<T5>::value, dummy, T5>::type T5_proc;
      typedef typename boost::conditional<stan::is_constant_struct<T6>::value, dummy, T6>::type T6_proc;
      
      size_t nvaris;
      agrad::vari** all_varis;
      T_partials_return* all_partials;

      vector_view_map<T1_proc, C1> d_x1;
      vector_view_map<T2_proc, C2> d_x2;
      vector_view_map<T3_proc, C3> d_x3;
      vector_view_map<T4_proc, C4> d_x4;
      vector_view_map<T5_proc, C5> d_x5;
      vector_view_map<T6_proc, C6> d_x6;

      OperandsAndPartials(const T1& x1 = 0, const T2& x2 = 0, const T3& x3 = 0,
                          const T4& x4 = 0, const T5& x5 = 0, const T6& x6 = 0)
        : nvaris(!is_constant_struct<T1>::value * length(x1) +
                 !is_constant_struct<T2>::value * length(x2) +
                 !is_constant_struct<T3>::value * length(x3) +
                 !is_constant_struct<T4>::value * length(x4) +
                 !is_constant_struct<T5>::value * length(x5) +
                 !is_constant_struct<T6>::value * length(x6)),
          all_varis(static_cast<agrad::vari**>
                    (agrad::chainable::operator new
                     (sizeof(agrad::vari*) * nvaris))),
          all_partials(static_cast<T_partials_return*>
                       (agrad::chainable::operator new
                        (sizeof(T_partials_return) * nvaris))),
          d_x1(x1, all_partials),
          d_x2(x2, all_partials
               + (!is_constant_struct<T1>::value) * length(x1)),
          d_x3(x3, all_partials
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)),
          d_x4(x4, all_partials
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)
               + (!is_constant_struct<T3>::value) * length(x3)),
          d_x5(x5, all_partials
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)
               + (!is_constant_struct<T3>::value) * length(x3)
               + (!is_constant_struct<T4>::value) * length(x4)),
          d_x6(x6, all_partials
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)
               + (!is_constant_struct<T3>::value) * length(x3)
               + (!is_constant_struct<T4>::value) * length(x4)
               + (!is_constant_struct<T5>::value) * length(x5)) {
        size_t base = 0;
        if (!is_constant_struct<T1>::value)
          base += set_varis<T1>().set(&all_varis[base], x1);
        if (!is_constant_struct<T2>::value)
          base += set_varis<T2>().set(&all_varis[base], x2);
        if (!is_constant_struct<T3>::value)
          base += set_varis<T3>().set(&all_varis[base], x3);
        if (!is_constant_struct<T4>::value)
          base += set_varis<T4>().set(&all_varis[base], x4);
        if (!is_constant_struct<T5>::value)
          base += set_varis<T5>().set(&all_varis[base], x5);
        if (!is_constant_struct<T6>::value)
          set_varis<T6>().set(&all_varis[base], x6);
        std::fill(all_partials, all_partials+nvaris, 0);
      }

      T_return_type
      to_var(T_partials_return logp) {
        return var(new agrad::partials_vari(logp, nvaris, all_varis,
                                              all_partials));
        }
    };
  }
}


using namespace Eigen;

int main() {
  using stan::agrad::var;
  Matrix<var, Dynamic, 1> m(10);
  VectorXd m_double(10);
  Matrix<var, Dynamic, Dynamic> l(10,10);
  MatrixXd l_double(10, 10);
  std::vector<Matrix<var, Dynamic, 1> > vecvec;
  vecvec.push_back(Matrix<var, Dynamic, 1>(10));
  vecvec.push_back(Matrix<var, Dynamic, 1>(10));
  std::vector<VectorXd> vecvec_double;
  vecvec_double.push_back(VectorXd(10));
  vecvec_double.push_back(VectorXd(10));
  double arrtest[10];
  double arrnull[10];
  double arrnull2[100];
  double arrnull3[20];
  double arrnull4[10];
  for (int i = 0; i < 10; ++i) {
    arrtest[i] = i;
    m(i) = arrtest[i];
    m_double(i) = arrtest[i];
  }
  vecvec[0] = m;
  vecvec[1] = m + m;
  vecvec_double[0] = m_double;
  vecvec_double[1] = m_double + m_double;
  for (int m = 0; m < 10; ++m) 
    for (int n = 0; n < 10; ++n) {
      l(m,n) = n * m;
      l_double(m,n) = m * n;
    }

  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << stan::agrad::ChainableStack::var_stack_.size() << std::endl;
  vector_view_map<Eigen::Matrix<stan::agrad::var, Eigen::Dynamic, 1>, Matrix<double, Dynamic, 1> > v(m, arrnull);
  vector_view_map<Eigen::Matrix<stan::agrad::var, Eigen::Dynamic, 1>, double> varray(m, arrnull4);
  vector_view_map<Eigen::Matrix<stan::agrad::var, Eigen::Dynamic, Eigen::Dynamic>, Matrix<double, Dynamic, Dynamic> > vl(l, arrnull2);
  vector_view_map<std::vector<Eigen::Matrix<stan::agrad::var, -1, 1> >, 
                              Eigen::Matrix<double, -1 ,1> > vvtest(vecvec, arrnull3);
  vector_view_map<boost::conditional<stan::is_constant_struct<double>::value, dummy, Matrix<double,Dynamic,1> >::type, Matrix<double, Dynamic, 1> > vt(m_double, arrnull3);
  stan::agrad::OperandsAndPartials<Eigen::Matrix<stan::agrad::var, Eigen::Dynamic, 1>, 
                                   Matrix<double, Dynamic, 1>,
                                   Eigen::Matrix<stan::agrad::var, Eigen::Dynamic, Dynamic>, 
                                   Matrix<double, Dynamic, Dynamic> > huge_test(m, l);
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << stan::agrad::ChainableStack::var_stack_.size() << std::endl;
  huge_test.d_x1[0] = m_double;
  huge_test.d_x1[1] = m_double;
  huge_test.d_x2[0] = l_double;
  std::cout << huge_test.d_x1[0] << std::endl;
  std::cout << huge_test.d_x2[0] << std::endl;
  huge_test.to_var(1.0); 
  std::cout << stan::agrad::ChainableStack::var_stack_.size() << std::endl;

  double ttest[1];
  vector_view_map<var, double> t(var(4.0), ttest); 
  v[0] = m_double;
  vl[0] = l_double.transpose();

  varray[0] = 1;
  varray[1] = 2;
  varray[2] = 1;
  varray[3] = 2;
  varray[4] = 1;
  varray[5] = 2;
  varray[6] = 1;
  varray[7] = 2;
  varray[8] = 1;
  varray[9] = 2;

  VectorXd zeros(10);
  zeros << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  std::cout << "Original" << vecvec[0] << std::endl;
  vvtest[0] = zeros;
  std::cout << "Pre += " << vvtest[0] << std::endl;
  vvtest[0] += vecvec_double[0];
  std::cout << "Post += " << vvtest[0] << std::endl;
  vvtest[1] = vecvec_double[1];

  for (int m = 0; m < 100; ++m) 
    std::cout << arrnull2[m] << std::endl;
  std::cout << "Orig" << std::endl;
  std::cout << vecvec[0] << std::endl;
  std::cout << vecvec[1] << std::endl;
  std::cout << "View" << std::endl;
  std::cout << vvtest[0] << std::endl;
  std::cout << vvtest[1] << std::endl;
  std::cout << "Raw array coming" << std::endl;
  for (int m = 0; m < 20; ++m)
    std::cout << arrnull3[m] << std::endl;
  std::cout << "Test changing one element" << std::endl;
  vvtest[1](1) = 1000;
  std::cout << vvtest[1] << std::endl;
  std::cout << "Raw array after change" << std::endl;
  for (int m = 0; m < 20; ++m)
    std::cout << arrnull3[m] << std::endl;

  for (int m = 0; m < 10; ++m) {
    std::cout << "vec_vew" << varray[m] - arrnull4[m] << std::endl;
    std::cout << v[0](m) << std::endl;
  }
}
