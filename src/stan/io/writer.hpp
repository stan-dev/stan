#ifndef __STAN__IO__WRITER_HPP__
#define __STAN__IO__WRITER_HPP__

#include <stdexcept>
#include <stan/prob/transform.hpp>

namespace stan {

  namespace io {

    /**
     * A stream-based writer for integer, scalar, vector, matrix
     * and array data types, which transforms from constrained to
     * a sequence of constrained variables.  
     *
     * <p>This class converts constrained values to unconstrained
     * values with mappings that invert those defined in
     * <code>stan::io::reader</code> to convert unconstrained values
     * to constrained values.
     *
     * @tparam T Basic scalar type.
     */
    template <typename T>
    class writer {
    private:
      std::vector<T> data_r_;
      std::vector<int> data_i_;
    public:

      typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> matrix_t;
      typedef Eigen::Matrix<T,Eigen::Dynamic,1> vector_t;
      typedef Eigen::Matrix<T,1,Eigen::Dynamic> row_vector_t;

      typedef Eigen::Array<T,Eigen::Dynamic,1> array_vec_t;

      /**
       * This is the tolerance for checking arithmetic bounds
       * in rank and in simplexes.  The current value is <code>1E-8</code>.
       */
      const double CONSTRAINT_TOLERANCE;

      /**
       * Construct a writer that writes to the specified
       * scalar and integer vectors.
       *
       * @param data_r Scalar values.
       * @param data_i Integer values.
       */
      writer(std::vector<T>& data_r,
             std::vector<int>& data_i)
        : data_r_(data_r),
          data_i_(data_i),
          CONSTRAINT_TOLERANCE(1E-8) {
        data_r_.clear();
        data_i_.clear();
      }

      /**
       * Destroy this writer.
       */
      ~writer() { }

      /**
       * Return a reference to the underlying vector of real values
       * that have been written.
       *
       * @return Values that have been written.
       */
      std::vector<T>& data_r() {
        return data_r_;
      }


      /**
       * Return a reference to the underlying vector of integer values
       * that have been written.
       *
       * @return Values that have been written.
       */
      std::vector<int>& data_i() {
        return data_i_;
      }

      /**
       * Write the specified integer to the sequence of integer values.
       *
       * @param n Integer to write.
       */
      void integer(int n) {
        data_i_.push_back(n);
      }

      /**
       * Write the unconstrained value corresponding to the specified
       * scalar.  Here, the unconstrain operation is a no-op, which
       * matches <code>reader::scalar_constrain()</code>.
       *
       * @param y The value.
       */
      void scalar_unconstrain(T& y) {
        data_r_.push_back(y);
      }

      /**
       * Write the unconstrained value corresponding to the specified
       * positive-constrained scalar.  The transformation applied is
       * <code>log(y)</code>, which is the inverse of constraining
       * transform specified in <code>reader::scalar_pos_constrain()</code>.
       *
       * <p>This method will fail if the argument is not non-negative.
       *
       * @param y The positive value.
       * @throw std::runtime_error if y is negative.
       */
      void scalar_pos_unconstrain(T& y) {
        if (y < 0.0)
          BOOST_THROW_EXCEPTION(std::runtime_error ("y is negative"));
        data_r_.push_back(log(y));
      }

      /**
       * Return the unconstrained version of the specified input,
       * which is constrained to be above the specified lower bound.
       * The unconstraining transform is <code>log(y - lb)</code>, which 
       * inverts the constraining
       * transform defined in <code>reader::scalar_lb_constrain(double)</code>,
       *
       * @param lb Lower bound.
       * @param y Lower-bounded value.
       * @throw std::runtime_error if y is lower than the lower bound provided.
       */
      void scalar_lb_unconstrain(double lb, T& y) {
        if (y < lb)
          BOOST_THROW_EXCEPTION(std::runtime_error ("y is lower than the lower bound"));
        data_r_.push_back(log(y - lb));
      }

      /**
       * Write the unconstrained value corresponding to the specified
       * lower-bounded value.  The unconstraining transform is
       * <code>log(ub - y)</code>, which reverses the constraining
       * transform defined in <code>reader::scalar_ub_constrain(double)</code>.
       *
       * @param ub Upper bound.
       * @param y Constrained value.
       * @throw std::runtime_error if y is higher than the upper bound provided.
       */
      void scalar_ub_unconstrain(double ub, T& y) {
        if (y > ub)
          BOOST_THROW_EXCEPTION(std::runtime_error ("y is higher than the lower bound"));
        data_r_.push_back(log(ub - y));
      }

      /**
       * Write the unconstrained value corresponding to the specified
       * value with the specified bounds.  The unconstraining
       * transform is given by <code>reader::logit((y-L)/(U-L))</code>, which
       * inverts the constraining transform defined in
       * <code>scalar_lub_constrain(double,double)</code>.
       *
       * @param lb Lower bound.
       * @param ub Upper bound.
       * @param y Bounded value.
       * @throw std::runtime_error if y is not between the lower and upper bounds
       */
      void scalar_lub_unconstrain(double lb, double ub, T& y) {
        if (y < lb || y > ub)
          BOOST_THROW_EXCEPTION(std::runtime_error ("y is not between the lower and upper bounds"));
        data_r_.push_back(stan::math::logit((y - lb) / (ub - lb)));
      }

      /**
       * Write the unconstrained value corresponding to the specified
       * correlation-constrained variable.
       *
       * <p>The unconstraining transform is <code>atanh(y)</code>, which
       * reverses the transfrom in <code>corr_constrain()</code>.
       *
       * @param y Correlation value.
       * @throw std::runtime_error if y is not between -1.0 and 1.0
       */
      void corr_unconstrain(T& y) {
        if (y > 1.0 || y < -1.0)
          BOOST_THROW_EXCEPTION(std::runtime_error ("y is not between -1.0 and 1.0"));
        data_r_.push_back(atanh(y));
      }

      /**
       * Write the unconstrained value corresponding to the
       * specified probability value.
       *
       * <p>The unconstraining transform is <code>logit(y)</code>,
       * which inverts the constraining transform defined in
       * <code>prob_constrain()</code>.
       *
       * @param y Probability value.
       * @throw std::runtime_error if y is not between 0.0 and 1.0
        */
      void prob_unconstrain(T& y) {
        if (y > 1.0 || y < 0.0)
          BOOST_THROW_EXCEPTION(std::runtime_error ("y is not between 0.0 and 1.0"));
        data_r_.push_back(stan::math::logit(y));
      }

      /**
       * Write the unconstrained vector that corresponds to the specified
       * ascendingly ordered vector.
       * 
       * <p>The unconstraining transform is defined for input vector <code>y</code>
       * to produce an output vector <code>x</code> of the same size, defined
       * by <code>x[0] = log(y[0])</code> and by
       * <code>x[k] = log(y[k] - y[k-1])</code> for <code>k > 0</code>.  This
       * unconstraining transform inverts the constraining transform specified
       * in <code>ordered_constrain(size_t)</code>.
       *
       * @param y Ascendingly ordered vector.
       * @return Unconstrained vector corresponding to the specified vector.
       * @throw std::runtime_error if vector is not in ascending order.
       */
      void ordered_unconstrain(vector_t& y) {
        if (y.size() == 0) return;
        stan::math::check_ordered("stan::io::ordered_unconstrain(%1%)", y, "Vector");
        data_r_.push_back(y[0]);
        for (typename vector_t::size_type i = 1; i < y.size(); ++i) {
          data_r_.push_back(log(y[i] - y[i-1]));
        }
      }

      /**
       * Write the unconstrained vector that corresponds to the specified
       * postiive ascendingly ordered vector.
       * 
       * <p>The unconstraining transform is defined for input vector <code>y</code>
       * to produce an output vector <code>x</code> of the same size, defined
       * by <code>x[0] = log(y[0])</code> and by
       * <code>x[k] = log(y[k] - y[k-1])</code> for <code>k > 0</code>.  This
       * unconstraining transform inverts the constraining transform specified
       * in <code>positive_ordered_constrain(size_t)</code>.
       *
       * @param y Positive ascendingly ordered vector.
       * @return Unconstrained vector corresponding to the specified vector.
       * @throw std::runtime_error if vector is not in ascending order.
       */
      void positive_ordered_unconstrain(vector_t& y) {
        if (y.size() == 0) return;
        stan::math::check_positive_ordered("stan::io::positive_ordered_unconstrain(%1%)", y, "Vector");
        data_r_.push_back(log(y[0]));
        for (typename vector_t::size_type i = 1; i < y.size(); ++i) {
          data_r_.push_back(log(y[i] - y[i-1]));
        }
      }


      /**
       * Write the specified unconstrained vector.
       * 
       * @param y Vector to write.
       */
      void vector_unconstrain(const vector_t& y) {
        for (typename vector_t::size_type i = 0; i < y.size(); ++i)
          data_r_.push_back(y[i]);
      }

      /**
       * Write the specified unconstrained vector.
       * 
       * @param y Vector to write.
       */
      void row_vector_unconstrain(const vector_t& y) {
        for (typename vector_t::size_type i = 0; i < y.size(); ++i)
          data_r_.push_back(y[i]);
      }

      /**
       * Write the specified unconstrained matrix.
       *
       * @param y Matrix to write.
       */
      void matrix_unconstrain(const matrix_t& y) {
        for (typename matrix_t::size_type j = 0; j < y.cols(); ++j) 
          for (typename matrix_t::size_type i = 0; i < y.rows(); ++i)
            data_r_.push_back(y(i,j));
      }

      void vector_lb_unconstrain(double lb, vector_t& y) {
        for (int i = 0; i < y.size(); ++i)
          scalar_lb_unconstrain(lb,y(i));
      }
      void row_vector_lb_unconstrain(double lb, row_vector_t& y) {
        for (int i = 0; i < y.size(); ++i)
          scalar_lb_unconstrain(lb,y(i));
      }
      void matrix_lb_unconstrain(double lb, matrix_t& y) {
        for (typename matrix_t::size_type j = 0; j < y.cols(); ++j) 
          for (typename matrix_t::size_type i = 0; i < y.rows(); ++i)
            scalar_lb_unconstrain(lb,y(i,j));
      }

      void vector_ub_unconstrain(double ub, vector_t& y) {
        for (int i = 0; i < y.size(); ++i)
          scalar_ub_unconstrain(ub,y(i));
      }
      void row_vector_ub_unconstrain(double ub, row_vector_t& y) {
        for (int i = 0; i < y.size(); ++i)
          scalar_ub_unconstrain(ub,y(i));
      }
      void matrix_ub_unconstrain(double ub, matrix_t& y) {
        for (typename matrix_t::size_type j = 0; j < y.cols(); ++j) 
          for (typename matrix_t::size_type i = 0; i < y.rows(); ++i)
            scalar_ub_unconstrain(ub,y(i,j));
      }


      void vector_lub_unconstrain(double lb, double ub, vector_t& y) {
        for (int i = 0; i < y.size(); ++i)
          scalar_lub_unconstrain(lb,ub,y(i));
      }
      void row_vector_lub_unconstrain(double lb, double ub, row_vector_t& y) {
        for (int i = 0; i < y.size(); ++i)
          scalar_lub_unconstrain(lb,ub,y(i));
      }
      void matrix_lub_unconstrain(double lb, double ub, matrix_t& y) {
        for (typename matrix_t::size_type j = 0; j < y.cols(); ++j) 
          for (typename matrix_t::size_type i = 0; i < y.rows(); ++i)
            scalar_lub_unconstrain(lb,ub,y(i,j));
      }

      

      /**
       * Write the unconstrained vector corresponding to the specified unit_vector 
       * value.  If the specified constrained unit_vector is of size <code>K</code>,
       * the returned unconstrained vector is of size <code>K-1</code>.
       *
       * <p>The transform takes <code>y = y[1],...,y[K]</code> and
       * produces the unconstrained vector. This inverts
       * the constraining transform of
       * <code>unit_vector_constrain(size_t)</code>.
       *
       * @param y Simplex constrained value.
       * @return Unconstrained value.
       * @throw std::runtime_error if the vector is not a unit_vector.
       */
      void unit_vector_unconstrain(vector_t& y) {
        stan::math::check_unit_vector("stan::io::unit_vector_unconstrain(%1%)", y, "Vector");
        vector_t uy = stan::prob::unit_vector_free(y);
        for (typename vector_t::size_type i = 0; i < uy.size(); ++i) 
          data_r_.push_back(uy[i]);
      }
 

      /**
       * Write the unconstrained vector corresponding to the specified simplex 
       * value.  If the specified constrained simplex is of size <code>K</code>,
       * the returned unconstrained vector is of size <code>K-1</code>.
       *
       * <p>The transform takes <code>y = y[1],...,y[K]</code> and
       * produces the unconstrained vector. This inverts
       * the constraining transform of
       * <code>simplex_constrain(size_t)</code>.
       *
       * @param y Simplex constrained value.
       * @return Unconstrained value.
       * @throw std::runtime_error if the vector is not a simplex.
       */
      void simplex_unconstrain(vector_t& y) {
        stan::math::check_simplex("stan::io::simplex_unconstrain(%1%)", y, "Vector");
        vector_t uy = stan::prob::simplex_free(y);
        for (typename vector_t::size_type i = 0; i < uy.size(); ++i) 
          data_r_.push_back(uy[i]);
      }

      /**
       * Writes the unconstrained Cholesky factor corresponding to the
       * specified constrained matrix.
       *
       * <p>The unconstraining operation is the inverse of the
       * constraining operation in
       * <code>cov_matrix_constrain(Matrix<T,Dynamic,Dynamic)</code>.
       *
       * @param y Constrained covariance matrix.
       * @throw std::runtime_error if y has no elements or if it is not square
       */
      void cholesky_factor_unconstrain(matrix_t& y) {
        // FIXME:  optimize by unrolling cholesky_factor_free
        Eigen::Matrix<T,Eigen::Dynamic,1> y_free
          = stan::prob::cholesky_factor_free(y);
        for (int i = 0; i < y_free.size(); ++i)
          data_r_.push_back(y_free[i]);
      }


      /**
       * Writes the unconstrained covariance matrix corresponding
       * to the specified constrained correlation matrix.
       *
       * <p>The unconstraining operation is the inverse of the
       * constraining operation in
       * <code>cov_matrix_constrain(Matrix<T,Dynamic,Dynamic)</code>.
       *
       * @param y Constrained covariance matrix.
       * @throw std::runtime_error if y has no elements or if it is not square
       */
      void cov_matrix_unconstrain(matrix_t& y) {
        typename matrix_t::size_type k = y.rows();
        if (k == 0 || y.cols() != k)
          BOOST_THROW_EXCEPTION(
              std::runtime_error ("y must have elements and y must be a square matrix"));
        typename matrix_t::size_type k_choose_2 = (k * (k-1)) / 2;
        array_vec_t cpcs(k_choose_2);
        array_vec_t sds(k);
        bool successful = stan::prob::factor_cov_matrix(cpcs,sds,y);
        if(!successful)
          BOOST_THROW_EXCEPTION(std::runtime_error ("factor_cov_matrix failed"));
        for (typename matrix_t::size_type i = 0; i < k_choose_2; ++i)
          data_r_.push_back(cpcs[i]);
        for (typename matrix_t::size_type i = 0; i < k; ++i)
          data_r_.push_back(sds[i]);
      }

      /**
       * Writes the unconstrained correlation matrix corresponding
       * to the specified constrained correlation matrix.
       *
       * <p>The unconstraining operation is the inverse of the
       * constraining operation in
       * <code>corr_matrix_constrain(Matrix<T,Dynamic,Dynamic)</code>.
       *
       * @param y Constrained correlation matrix.
       * @throw std::runtime_error if the correlation matrix has no elements or
       *    is not a square matrix.
       * @throw std::runtime_error if the correlation matrix cannot be factorized
       *    by factor_cov_matrix() or if the sds returned by factor_cov_matrix()
       *    on log scale are unconstrained.
       */
      void corr_matrix_unconstrain(matrix_t& y) {
        stan::math::check_corr_matrix("stan::io::corr_matrix_unconstrain(%1%)", y, "Matrix");
        size_t k = y.rows();
        size_t k_choose_2 = (k * (k-1)) / 2;
        array_vec_t cpcs(k_choose_2);
        array_vec_t sds(k);
        bool successful = stan::prob::factor_cov_matrix(cpcs,sds,y);
        if (!successful)
          BOOST_THROW_EXCEPTION(std::runtime_error ("y cannot be factorized by factor_cov_matrix"));
        for (size_t i = 0; i < k; ++i) {
          // sds on log scale unconstrained
          if (fabs(sds[i] - 0.0) >= CONSTRAINT_TOLERANCE)
            BOOST_THROW_EXCEPTION(std::runtime_error ("sds on log scale are unconstrained"));
        }
        for (size_t i = 0; i < k_choose_2; ++i)
          data_r_.push_back(cpcs[i]);
      }

    };
  }

}

#endif
