#ifndef STAN__PROB__TRANSFORM_HPP
#define STAN__PROB__TRANSFORM_HPP

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <boost/throw_exception.hpp>
#include <boost/math/tools/promotion.hpp>

#include <stan/math.hpp>
#include <stan/error_handling/scalar/check_bounded.hpp>
#include <stan/error_handling/scalar/check_greater_or_equal.hpp>
#include <stan/error_handling/matrix/check_square.hpp>
#include <stan/math/matrix.hpp>

#include <stan/math/matrix/sum.hpp>
#include <stan/error_handling/matrix.hpp>
#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>
#include <stan/math/matrix/meta/index_type.hpp>


namespace stan {
  
  namespace prob {


    const double CONSTRAINT_TOLERANCE = 1E-8;

    /**
     * This function is intended to make starting values, given a unit
     * upper-triangular matrix U such that U'DU is a correlation matrix
     *   
     * @param CPCs fill this unbounded
     * @param Sigma U matrix
     */
    template<typename T>
    void    
    factor_U(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& U,
             Eigen::Array<T,Eigen::Dynamic,1>& CPCs) { 

      size_t K = U.rows();
      size_t position = 0;
      size_t pull = K - 1;

      if (K == 2) {
        CPCs(0) = atanh(U(0,1));
        return;
      }

      Eigen::Array<T,1,Eigen::Dynamic> temp = U.row(0).tail(pull);

      CPCs.head(pull) = temp;

      Eigen::Array<T,Eigen::Dynamic,1> acc(K);
      acc(0) = -0.0;
      acc.tail(pull) = 1.0 - temp.square();
      for(size_t i = 1; i < (K - 1); i++) {
        position += pull;
        pull--;
        temp = U.row(i).tail(pull);
        temp /= sqrt(acc.tail(pull) / acc(i));
        CPCs.segment(position, pull) = temp;
        acc.tail(pull) *= 1.0 - temp.square();
      }
      CPCs = 0.5 * ( (1.0 + CPCs) / (1.0 - CPCs) ).log(); // now unbounded
    }
    

    /**
     * This function is intended to make starting values, given a
     * covariance matrix Sigma
     *
     * The transformations are hard coded as log for standard
     * deviations and Fisher transformations (atanh()) of CPCs
     *
     * @param[in] Sigma covariance matrix
     * @param[out] CPCs fill this unbounded (does not resize)
     * @param[out] sds fill this unbounded (does not resize)
     * @return false if any of the diagonals of Sigma are 0
     */
    template<typename T>
    bool
    factor_cov_matrix(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                      Eigen::Array<T,Eigen::Dynamic,1>& CPCs,
                      Eigen::Array<T,Eigen::Dynamic,1>& sds) {

      size_t K = sds.rows();

      sds = Sigma.diagonal().array();
      if( (sds <= 0.0).any() ) return false;
      sds = sds.sqrt();

      Eigen::DiagonalMatrix<T,Eigen::Dynamic> D(K);
      D.diagonal() = sds.inverse();
      sds = sds.log(); // now unbounded

      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> R = D * Sigma * D;
      // to hopefully prevent pivoting due to floating point error
      R.diagonal().setOnes(); 
      Eigen::LDLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > ldlt;
      ldlt = R.ldlt();
      if (!ldlt.isPositive()) 
        return false;
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> U = ldlt.matrixU();
      factor_U(U, CPCs);
      return true;
    }

    // MATRIX TRANSFORMS +/- JACOBIANS

    /**
     * Return the Cholesky factor of the correlation matrix of the
     * specified dimensionality corresponding to the specified
     * canonical partial correlations.
     * 
     * It is generally better to work with the Cholesky factor rather
     * than the correlation matrix itself when the determinant,
     * inverse, etc. of the correlation matrix is needed for some
     * statistical calculation.
     *
     * <p>See <code>read_corr_matrix(Array,size_t,T)</code>
     * for more information.
     *
     * @param CPCs The (K choose 2) canonical partial correlations in
     * (-1,1).
     * @param K Dimensionality of correlation matrix.
     * @return Cholesky factor of correlation matrix for specified
     * canonical partial correlations.

     * @tparam T Type of underlying scalar.  
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_corr_L(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs, // on (-1,1)
                const size_t K) {
      Eigen::Array<T,Eigen::Dynamic,1> temp;         
      Eigen::Array<T,Eigen::Dynamic,1> acc(K-1);  
      acc.setOnes();
      // Cholesky factor of correlation matrix
      Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic> L(K,K); 
      L.setZero();

      size_t position = 0;
      size_t pull = K - 1;

      L(0,0) = 1.0;
      L.col(0).tail(pull) = temp = CPCs.head(pull);
      acc.tail(pull) = 1.0 - temp.square();
      for(size_t i = 1; i < (K - 1); i++) {
        position += pull;
        pull--;
        temp = CPCs.segment(position, pull);
        L(i,i) = sqrt(acc(i-1));
        L.col(i).tail(pull) = temp * acc.tail(pull).sqrt();
        acc.tail(pull) *= 1.0 - temp.square();
      }
      L(K-1,K-1) = sqrt(acc(K-2));
      return L.matrix();
    }

    /**
     * Return the correlation matrix of the specified dimensionality 
     * corresponding to the specified canonical partial correlations.
     *
     * <p>See <code>read_corr_matrix(Array,size_t,T)</code>
     * for more information.
     *
     * @param CPCs The (K choose 2) canonical partial correlations in (-1,1).
     * @param K Dimensionality of correlation matrix.
     * @return Cholesky factor of correlation matrix for specified
     * canonical partial correlations.
     * @tparam T Type of underlying scalar.  
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_corr_matrix(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs, 
                     const size_t K) {
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L 
        = read_corr_L(CPCs, K);
      using stan::math::multiply_lower_tri_self_transpose;
      return multiply_lower_tri_self_transpose(L);
    }
    
    /**
     * Return the Cholesky factor of the correlation matrix of the
     * specified dimensionality corresponding to the specified
     * canonical partial correlations, incrementing the specified
     * scalar reference with the log absolute determinant of the
     * Jacobian of the transformation.
     *
     * <p>The implementation is Ben Goodrich's Cholesky
     * factor-based approach to the C-vine method of:
     * 
     * <ul><li> Daniel Lewandowski, Dorota Kurowicka, and Harry Joe,
     * Generating random correlation matrices based on vines and
     * extended onion method Journal of Multivariate Analysis 100
     * (2009) 1989–2001 </li></ul>
     *
     * @param CPCs The (K choose 2) canonical partial correlations in
     * (-1,1).
     * @param K Dimensionality of correlation matrix.
     * @param log_prob Reference to variable to increment with the log
     * Jacobian determinant.
     * @return Cholesky factor of correlation matrix for specified
     * partial correlations.
     * @tparam T Type of underlying scalar.  
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_corr_L(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs,
                const size_t K,
                T& log_prob) {

      using stan::math::log1m;
      using stan::math::square;
      using stan::math::sum;

      Eigen::Matrix<T,Eigen::Dynamic,1> values(CPCs.rows() - 1);
      size_t pos = 0;
      // no need to abs() because this Jacobian determinant 
      // is strictly positive (and triangular)
      // see inverse of Jacobian in equation 11 of LKJ paper
      for (size_t k = 1; k <= (K - 2); k++)
        for (size_t i = k + 1; i <= K; i++) {
          values(pos) = (K - k - 1) * log1m(square(CPCs(pos)));
          pos++;
        }

      log_prob += 0.5 * sum(values);
      return read_corr_L(CPCs,K);
    }

    /**
     * Return the correlation matrix of the specified dimensionality
     * corresponding to the specified canonical partial correlations,
     * incrementing the specified scalar reference with the log
     * absolute determinant of the Jacobian of the transformation.
     *
     * It is usually preferable to utilize the version that returns
     * the Cholesky factor of the correlation matrix rather than the
     * correlation matrix itself in statistical calculations.
     * 
     * @param CPCs The (K choose 2) canonical partial correlations in
     * (-1,1).
     * @param K Dimensionality of correlation matrix.
     * @param log_prob Reference to variable to increment with the log
     * Jacobian determinant.
     * @return Correlation matrix for specified partial correlations.
     * @tparam T Type of underlying scalar.  
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_corr_matrix(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs,
                     const size_t K,
                     T& log_prob) {

      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L 
        = read_corr_L(CPCs, K, log_prob);
      using stan::math::multiply_lower_tri_self_transpose;
      return multiply_lower_tri_self_transpose(L);
    }
    
    /** 
     * This is the function that should be called prior to evaluating
     * the density of any elliptical distribution
     *
     * @param CPCs on (-1,1)
     * @param sds on (0,inf)
     * @param log_prob the log probability value to increment with the Jacobian
     * @return Cholesky factor of covariance matrix for specified
     * partial correlations.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_cov_L(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs,
               const Eigen::Array<T,Eigen::Dynamic,1>& sds, 
               T& log_prob) {
      size_t K = sds.rows();
      // adjust due to transformation from correlations to covariances
      log_prob += (sds.log().sum() + stan::math::LOG_2) * K;
      return sds.matrix().asDiagonal() * read_corr_L(CPCs, K, log_prob);
    }

    /** 
     * A generally worse alternative to call prior to evaluating the
     * density of an elliptical distribution
     *
     * @param CPCs on (-1,1)
     * @param sds on (0,inf)
     * @param log_prob the log probability value to increment with the Jacobian
     * @return Covariance matrix for specified partial correlations.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_cov_matrix(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs,
                    const Eigen::Array<T,Eigen::Dynamic,1>& sds, 
                    T& log_prob) {

      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L 
        = read_cov_L(CPCs, sds, log_prob);
      using stan::math::multiply_lower_tri_self_transpose;
      return multiply_lower_tri_self_transpose(L);
    }

    /** 
     *
     * Builds a covariance matrix from CPCs and standard deviations
     *
     * @param CPCs in (-1,1)
     * @param sds in (0,inf)
     */
    template<typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_cov_matrix(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs, 
                    const Eigen::Array<T,Eigen::Dynamic,1>& sds) {

      size_t K = sds.rows();
      Eigen::DiagonalMatrix<T,Eigen::Dynamic> D(K);
      D.diagonal() = sds;
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L 
        = D * read_corr_L(CPCs, K);
      using stan::math::multiply_lower_tri_self_transpose;
      return multiply_lower_tri_self_transpose(L);
    }


    /** 
     * This function calculates the degrees of freedom for the t
     * distribution that corresponds to the shape parameter in the
     * Lewandowski et. al. distribution 
     *
     * @param eta hyperparameter on (0,inf), eta = 1 <-> correlation
     * matrix is uniform
     * @param K number of variables in covariance matrix
     */
    template<typename T>
    const Eigen::Array<T,Eigen::Dynamic,1>
    make_nu(const T eta, const size_t K) {
      using Eigen::Array;
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;
  
      Array<T,Dynamic,1> nu(K * (K - 1) / 2);
  
      T alpha = eta + (K - 2.0) / 2.0; // from Lewandowski et. al.

      // Best (1978) implies nu = 2 * alpha for the dof in a t 
      // distribution that generates a beta variate on (-1,1)
      T alpha2 = 2.0 * alpha; 
      for (size_type j = 0; j < (K - 1); j++) {
        nu(j) = alpha2;
      }
      size_t counter = K - 1;
      for (size_type i = 1; i < (K - 1); i++) {
        alpha -= 0.5;
        alpha2 = 2.0 * alpha;
        for (size_type j = i + 1; j < K; j++) {
          nu(counter) = alpha2;
          counter++;
        }
      }
      return nu;
    }




    // IDENTITY

    /**
     * Returns the result of applying the identity constraint
     * transform to the input.
     *
     * <p>This method is effectively a no-op and is mainly useful as a
     * placeholder in auto-generated code.
     *
     * @param x Free scalar.
     * @return Transformed input.
     *
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline 
    T identity_constrain(T x) {
      return x;
    }

    /**
     * Returns the result of applying the identity constraint
     * transform to the input and increments the log probability
     * reference with the log absolute Jacobian determinant.
     *
     * <p>This method is effectively a no-op and mainly useful as a
     * placeholder in auto-generated code.
     *
     * @param x Free scalar.
     * lp Reference to log probability.
     * @return Transformed input.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline 
    T identity_constrain(const T x, T& /*lp*/) {
      return x;
    }
    
    /**
     * Returns the result of applying the inverse of the identity
     * constraint transform to the input.
     *
     * <p>This method is effectively a no-op and mainly useful as a
     * placeholder in auto-generated code.
     *
     * @param y Constrained scalar.
     * @return The input.
     * @tparam T Type of scalar.
     */
    template <typename T> 
    inline
    T identity_free(const T y) {
      return y;
    }


    // POSITIVE

    /**
     * Return the positive value for the specified unconstrained input.
     *
     * <p>The transform applied is
     *
     * <p>\f$f(x) = \exp(x)\f$.
     * 
     * @param x Arbitrary input scalar.
     * @return Input transformed to be positive.
     */
    template <typename T> 
    inline
    T positive_constrain(const T x) {
      return exp(x);
    }

    /**
     * Return the positive value for the specified unconstrained input,
     * incrementing the scalar reference with the log absolute
     * Jacobian determinant.
     *
     * <p>See <code>positive_constrain(T)</code> for details
     * of the transform.  The log absolute Jacobian determinant is
     *
     * <p>\f$\log | \frac{d}{dx} \mbox{exp}(x) | 
     *    = \log | \mbox{exp}(x) | =  x\f$.
     * 
     * @param x Arbitrary input scalar.
     * @param lp Log probability reference.
     * @return Input transformed to be positive.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T positive_constrain(const T x, T& lp) {
      lp += x;
      return exp(x); 
    }

    /**
     * Return the unconstrained value corresponding to the specified
     * positive-constrained value.  
     *
     * <p>The transform is the inverse of the transform \f$f\f$ applied by
     * <code>positive_constrain(T)</code>, namely
     *
     * <p>\f$f^{-1}(x) = \log(x)\f$.
     * 
     * <p>The input is validated using <code>stan::error_handling::check_positive()</code>.
     * 
     * @param y Input scalar.
     * @return Unconstrained value that produces the input when constrained.
     * @tparam T Type of scalar.
     * @throw std::domain_error if the variable is negative.
     */
    template <typename T>
    inline
    T positive_free(const T y) {
      stan::error_handling::check_positive("stan::prob::positive_free", "Positive variable", y);
      return log(y);
    }

    // LOWER BOUND

    /**
     * Return the lower-bounded value for the specified unconstrained input
     * and specified lower bound.
     *
     * <p>The transform applied is
     *
     * <p>\f$f(x) = \exp(x) + L\f$
     *
     * <p>where \f$L\f$ is the constant lower bound.
     *
     * <p>If the lower bound is negative infinity, this function
     * reduces to <code>identity_constrain(x)</code>.
     *
     * @param x Unconstrained scalar input.
     * @param lb Lower-bound on constrained ouptut.
     * @return Lower-bound constrained value correspdonding to inputs.
     * @tparam T Type of scalar.
     * @tparam TL Type of lower bound.
     */
    template <typename T, typename TL>
    inline
    T lb_constrain(const T x, const TL lb) {
      if (lb == -std::numeric_limits<double>::infinity())
        return identity_constrain(x);
      return exp(x) + lb;
    }

    /**
     * Return the lower-bounded value for the speicifed unconstrained
     * input and specified lower bound, incrementing the specified
     * reference with the log absolute Jacobian determinant of the
     * transform.
     *
     * If the lower bound is negative infinity, this function
     * reduces to <code>identity_constraint(x,lp)</code>.
     *
     * @param x Unconstrained scalar input.
     * @param lb Lower-bound on output.
     * @param lp Reference to log probability to increment.
     * @return Loer-bound constrained value corresponding to inputs.
     * @tparam T Type of scalar.
     * @tparam TL Type of lower bound.
     */
    template <typename T, typename TL>
    inline
    typename boost::math::tools::promote_args<T,TL>::type
    lb_constrain(const T x, const TL lb, T& lp) {
      if (lb == -std::numeric_limits<double>::infinity())
        return identity_constrain(x,lp);
      lp += x;
      return exp(x) + lb;
    }

    /**
     * Return the unconstrained value that produces the specified
     * lower-bound constrained value.
     *
     * If the lower bound is negative infinity, it is ignored and
     * the function reduces to <code>identity_free(y)</code>.
     * 
     * @param y Input scalar.
     * @param lb Lower bound.
     * @return Unconstrained value that produces the input when
     * constrained.
     * @tparam T Type of scalar.
     * @tparam TL Type of lower bound.
     * @throw std::domain_error if y is lower than the lower bound.
     */
    template <typename T, typename TL>
    inline
    typename boost::math::tools::promote_args<T,TL>::type
    lb_free(const T y, const TL lb) {
      if (lb == -std::numeric_limits<double>::infinity())
        return identity_free(y);
      stan::error_handling::check_greater_or_equal("stan::prob::lb_free", 
                                                   "Lower bounded variable", y, lb);
      return log(y - lb);
    }
    

    // UPPER BOUND

    /**
     * Return the upper-bounded value for the specified unconstrained
     * scalar and upper bound.
     *
     * <p>The transform is
     *
     * <p>\f$f(x) = U - \exp(x)\f$
     *
     * <p>where \f$U\f$ is the upper bound.  
     * 
     * If the upper bound is positive infinity, this function
     * reduces to <code>identity_constrain(x)</code>.
     * 
     * @param x Free scalar.
     * @param ub Upper bound.
     * @return Transformed scalar with specified upper bound.
     * @tparam T Type of scalar.
     * @tparam TU Type of upper bound.
     */
    template <typename T, typename TU>
    inline
    typename boost::math::tools::promote_args<T,TU>::type
    ub_constrain(const T x, const TU ub) {
      if (ub == std::numeric_limits<double>::infinity())
        return identity_constrain(x);
      return ub - exp(x);
    }

    /**
     * Return the upper-bounded value for the specified unconstrained
     * scalar and upper bound and increment the specified log
     * probability reference with the log absolute Jacobian
     * determinant of the transform.
     *
     * <p>The transform is as specified for
     * <code>ub_constrain(T,double)</code>.  The log absolute Jacobian
     * determinant is
     *
     * <p>\f$ \log | \frac{d}{dx} -\mbox{exp}(x) + U | 
     *     = \log | -\mbox{exp}(x) + 0 | = x\f$.
     *
     * If the upper bound is positive infinity, this function
     * reduces to <code>identity_constrain(x,lp)</code>.
     *
     * @param x Free scalar.
     * @param ub Upper bound.
     * @param lp Log probability reference.
     * @return Transformed scalar with specified upper bound.
     * @tparam T Type of scalar.
     * @tparam TU Type of upper bound.
     */
    template <typename T, typename TU>
    inline
    typename boost::math::tools::promote_args<T,TU>::type
    ub_constrain(const T x, const TU ub, T& lp) {
      if (ub == std::numeric_limits<double>::infinity())
        return identity_constrain(x,lp);
      lp += x;
      return ub - exp(x);
    }

    /**
     * Return the free scalar that corresponds to the specified
     * upper-bounded value with respect to the specified upper bound.
     *
     * <p>The transform is the reverse of the
     * <code>ub_constrain(T,double)</code> transform,
     *
     * <p>\f$f^{-1}(y) = \log -(y - U)\f$
     *
     * <p>where \f$U\f$ is the upper bound.
     *
     * If the upper bound is positive infinity, this function
     * reduces to <code>identity_free(y)</code>.
     *
     * @param y Upper-bounded scalar.
     * @param ub Upper bound.
     * @return Free scalar corresponding to upper-bounded scalar.
     * @tparam T Type of scalar.
     * @tparam TU Type of upper bound.
     * @throw std::invalid_argument if y is greater than the upper
     * bound.
     */
    template <typename T, typename TU>
    inline
    typename boost::math::tools::promote_args<T,TU>::type
    ub_free(const T y, const TU ub) {
      if (ub == std::numeric_limits<double>::infinity())
        return identity_free(y);
      stan::error_handling::check_less_or_equal("stan::prob::ub_free", 
                                                "Upper bounded variable", y, ub);
      return log(ub - y);
    }


    // LOWER & UPPER BOUNDS

    /**
     * Return the lower- and upper-bounded scalar derived by
     * transforming the specified free scalar given the specified
     * lower and upper bounds.
     *
     * <p>The transform is the transformed and scaled inverse logit,
     *
     * <p>\f$f(x) = L + (U - L) \mbox{logit}^{-1}(x)\f$
     *
     * If the lower bound is negative infinity and upper bound finite,
     * this function reduces to <code>ub_constrain(x,ub)</code>.  If
     * the upper bound is positive infinity and the lower bound
     * finite, this function reduces to
     * <code>lb_constrain(x,lb)</code>.  If the upper bound is
     * positive infinity and the lower bound negative infinity, 
     * this function reduces to <code>identity_constrain(x)</code>.
     * 
     * @param x Free scalar to transform.
     * @param lb Lower bound.
     * @param ub Upper bound.
     * @return Lower- and upper-bounded scalar derived from transforming
     * the free scalar.
     * @tparam T Type of scalar.
     * @tparam TL Type of lower bound.
     * @tparam TU Type of upper bound.
     * @throw std::domain_error if ub <= lb
     */
    template <typename T, typename TL, typename TU>
    inline
    typename boost::math::tools::promote_args<T,TL,TU>::type
    lub_constrain(const T x, TL lb, TU ub) {
      stan::error_handling::check_less("lub_constrain", "lb", lb, ub);
      if (lb == -std::numeric_limits<double>::infinity())
        return ub_constrain(x,ub);
      if (ub == std::numeric_limits<double>::infinity())
        return lb_constrain(x,lb);

      T inv_logit_x;
      if (x > 0) {
        T exp_minus_x = exp(-x);
        inv_logit_x = 1.0 / (1.0 + exp_minus_x);
        // Prevent x from reaching one unless it really really should.
        if ((x < std::numeric_limits<double>::infinity()) 
            && (inv_logit_x == 1))
            inv_logit_x = 1 - 1e-15;
      } else {
        T exp_x = exp(x);
        inv_logit_x = 1.0 - 1.0 / (1.0 + exp_x);
        // Prevent x from reaching zero unless it really really should.
        if ((x > -std::numeric_limits<double>::infinity()) 
            && (inv_logit_x== 0))
            inv_logit_x = 1e-15;
      }
      return lb + (ub - lb) * inv_logit_x;
    }

    /**
     * Return the lower- and upper-bounded scalar derived by
     * transforming the specified free scalar given the specified
     * lower and upper bounds and increment the specified log
     * probability with the log absolute Jacobian determinant.
     *
     * <p>The transform is as defined in
     * <code>lub_constrain(T,double,double)</code>.  The log absolute
     * Jacobian determinant is given by
     * 
     * <p>\f$\log \left| \frac{d}{dx} \left(
     *                L + (U-L) \mbox{logit}^{-1}(x) \right) 
     *            \right|\f$
     *
     * <p>\f$ {} = \log |
     *         (U-L)
     *         \, (\mbox{logit}^{-1}(x)) 
     *         \, (1 - \mbox{logit}^{-1}(x)) |\f$
     *
     * <p>\f$ {} = \log (U - L) + \log (\mbox{logit}^{-1}(x)) 
     *                          + \log (1 - \mbox{logit}^{-1}(x))\f$
     *
     * <p>If the lower bound is negative infinity and upper bound finite,
     * this function reduces to <code>ub_constrain(x,ub,lp)</code>.  If
     * the upper bound is positive infinity and the lower bound
     * finite, this function reduces to
     * <code>lb_constrain(x,lb,lp)</code>.  If the upper bound is
     * positive infinity and the lower bound negative infinity, 
     * this function reduces to <code>identity_constrain(x,lp)</code>.
     *
     * @param x Free scalar to transform.
     * @param lb Lower bound.
     * @param ub Upper bound.
     * @param lp Log probability scalar reference.
     * @return Lower- and upper-bounded scalar derived from transforming
     * the free scalar.
     * @tparam T Type of scalar.
     * @tparam TL Type of lower bound.
     * @tparam TU Type of upper bound.
     * @throw std::domain_error if ub <= lb
     */
    template <typename T, typename TL, typename TU>
    typename boost::math::tools::promote_args<T,TL,TU>::type
    lub_constrain(const T x, const TL lb, const TU ub, T& lp) {
      if (!(lb < ub)) {
        std::stringstream s;
        s << "domain error in lub_constrain;  lower bound = " << lb
          << " must be strictly less than upper bound = " << ub;
        throw std::domain_error(s.str());
      }
      if (lb == -std::numeric_limits<double>::infinity())
        return ub_constrain(x,ub,lp);
      if (ub == std::numeric_limits<double>::infinity())
        return lb_constrain(x,lb,lp);
      T inv_logit_x;
      if (x > 0) {
        T exp_minus_x = exp(-x);
        inv_logit_x = 1.0 / (1.0 + exp_minus_x);
        lp += log(ub - lb) - x - 2 * log1p(exp_minus_x);
        // Prevent x from reaching one unless it really really should.
        if ((x < std::numeric_limits<double>::infinity()) 
            && (inv_logit_x == 1))
            inv_logit_x = 1 - 1e-15;
      } else {
        T exp_x = exp(x);
        inv_logit_x = 1.0 - 1.0 / (1.0 + exp_x);
        lp += log(ub - lb) + x - 2 * log1p(exp_x);
        // Prevent x from reaching zero unless it really really should.
        if ((x > -std::numeric_limits<double>::infinity()) 
            && (inv_logit_x== 0))
            inv_logit_x = 1e-15;
      }
      return lb + (ub - lb) * inv_logit_x;
    }

    /**
     * Return the unconstrained scalar that transforms to the
     * specified lower- and upper-bounded scalar given the specified
     * bounds.
     *
     * <p>The transfrom in <code>lub_constrain(T,double,double)</code>, 
     * is reversed by a transformed and scaled logit,
     *
     * <p>\f$f^{-1}(y) = \mbox{logit}(\frac{y - L}{U - L})\f$
     *
     * where \f$U\f$ and \f$L\f$ are the lower and upper bounds.
     *
     * <p>If the lower bound is negative infinity and upper bound finite,
     * this function reduces to <code>ub_free(y,ub)</code>.  If
     * the upper bound is positive infinity and the lower bound
     * finite, this function reduces to
     * <code>lb_free(x,lb)</code>.  If the upper bound is
     * positive infinity and the lower bound negative infinity, 
     * this function reduces to <code>identity_free(y)</code>.
     *
     * @tparam T Type of scalar.
     * @param y Scalar input.
     * @param lb Lower bound.
     * @param ub Upper bound.
     * @return The free scalar that transforms to the input scalar
     * given the bounds.
     * @throw std::invalid_argument if the lower bound is greater than
     *   the upper bound, y is less than the lower bound, or y is
     *   greater than the upper bound
     */
    template <typename T, typename TL, typename TU>
    inline
    typename boost::math::tools::promote_args<T,TL,TU>::type
    lub_free(const T y, TL lb, TU ub) {
      using stan::math::logit;
      stan::error_handling::check_bounded<T, TL, TU>
        ("stan::prob::lub_free",
         "Bounded variable",
         y, lb, ub);
      if (lb == -std::numeric_limits<double>::infinity())
        return ub_free(y,ub);
      if (ub == std::numeric_limits<double>::infinity())
        return lb_free(y,lb);
      return logit((y - lb) / (ub - lb));
    }

    
    // PROBABILITY

    /**
     * Return a probability value constrained to fall between 0 and 1
     * (inclusive) for the specified free scalar.
     *
     * <p>The transform is the inverse logit,
     *
     * <p>\f$f(x) = \mbox{logit}^{-1}(x) = \frac{1}{1 + \exp(x)}\f$.
     *
     * @param x Free scalar.
     * @return Probability-constrained result of transforming
     * the free scalar.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T prob_constrain(const T x) {
      using stan::math::inv_logit;
      return inv_logit(x);
    }

    /**
     * Return a probability value constrained to fall between 0 and 1
     * (inclusive) for the specified free scalar and increment the
     * specified log probability reference with the log absolute Jacobian
     * determinant of the transform.
     *
     * <p>The transform is as defined for <code>prob_constrain(T)</code>. 
     * The log absolute Jacobian determinant is
     *
     * <p>The log absolute Jacobian determinant is 
     *
     * <p>\f$\log | \frac{d}{dx} \mbox{logit}^{-1}(x) |\f$
     * <p>\f$\log ((\mbox{logit}^{-1}(x)) (1 - \mbox{logit}^{-1}(x))\f$
     * <p>\f$\log (\mbox{logit}^{-1}(x)) + \log (1 - \mbox{logit}^{-1}(x))\f$.
     *
     * @param x Free scalar.
     * @param lp Log probability reference.
     * @return Probability-constrained result of transforming
     * the free scalar.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T prob_constrain(const T x, T& lp) {
      using stan::math::inv_logit;
      using stan::math::log1m;
      T inv_logit_x = inv_logit(x);
      lp += log(inv_logit_x) + log1m(inv_logit_x);
      return inv_logit_x;
    }

    /**
     * Return the free scalar that when transformed to a probability
     * produces the specified scalar.
     *
     * <p>The function that reverses the constraining transform
     * specified in <code>prob_constrain(T)</code> is the logit
     * function,
     *
     * <p>\f$f^{-1}(y) = \mbox{logit}(y) = \frac{1 - y}{y}\f$.
     * 
     * @param y Scalar input.
     * @tparam T Type of scalar.
     * @throw std::domain_error if y is less than 0 or greater than 1.
     */
    template <typename T>
    inline
    T prob_free(const T y) {
      using stan::math::logit;
      stan::error_handling::check_bounded<T,double,double>
        ("stan::prob::prob_free", "Probability variable",
         y, 0, 1);
      return logit(y);
    }
    
    
    // CORRELATION

    /**
     * Return the result of transforming the specified scalar to have
     * a valid correlation value between -1 and 1 (inclusive).
     *
     * <p>The transform used is the hyperbolic tangent function,
     *
     * <p>\f$f(x) = \tanh x = \frac{\exp(2x) - 1}{\exp(2x) + 1}\f$.
     * 
     * @param x Scalar input.
     * @return Result of transforming the input to fall between -1 and 1.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T corr_constrain(const T x) {
      return tanh(x);
    }

    /**
     * Return the result of transforming the specified scalar to have
     * a valid correlation value between -1 and 1 (inclusive).
     *
     * <p>The transform used is as specified for
     * <code>corr_constrain(T)</code>.  The log absolute Jacobian
     * determinant is
     *
     * <p>\f$\log | \frac{d}{dx} \tanh x  | = \log (1 - \tanh^2 x)\f$.
     * 
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T corr_constrain(const T x, T& lp) {
      using stan::math::log1m;
      T tanh_x = tanh(x);
      lp += log1m(tanh_x * tanh_x);
      return tanh_x;
    }

    /**
     * Return the unconstrained scalar that when transformed to
     * a valid correlation produces the specified value.
     *
     * <p>This function inverts the transform defined for
     * <code>corr_constrain(T)</code>, which is the inverse hyperbolic
     * tangent,
     *
     * <p>\f$ f^{-1}(y)
     *          = \mbox{atanh}\, y
     *          = \frac{1}{2} \log \frac{y + 1}{y - 1}\f$.
     *
     * @param y Correlation scalar input.
     * @return Free scalar that transforms to the specified input.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T corr_free(const T y) {
      stan::error_handling::check_bounded<T,double,double>
        ("stan::prob::lub_free",
         "Correlation variable", y, -1, 1);
      return atanh(y);
    }


    // Unit vector   

    /**
     * Return the unit length vector corresponding to the free vector y.
     * The free vector contains K-1 spherical coordinates.
     *
     * @param y of K - 1 spherical coordinates
     * @return Unit length vector of dimension K
     * @tparam T Scalar type.
     **/
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    unit_vector_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& y) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;
      int Km1 = y.size();
      Matrix<T,Dynamic,1> x(Km1 + 1);
      x(0) = 1.0;
      const T half_pi = T(M_PI/2.0);
      for (size_type k = 1; k <= Km1; ++k) {
        T yk_1 = y(k-1) + half_pi;
        T sin_yk_1 = sin(yk_1);
        x(k) = x(k-1)*sin_yk_1; 
        x(k-1) *= cos(yk_1);
      }
      return x;
    }

    /**
     * Return the unit length vector corresponding to the free vector y.
     * The free vector contains K-1 spherical coordinates.
     *
     * @param y of K - 1 spherical coordinates
     * @return Unit length vector of dimension K
     * @param lp Log probability reference to increment.
     * @tparam T Scalar type.
     **/
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    unit_vector_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& y, T &lp) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      int Km1 = y.size();
      Matrix<T,Dynamic,1> x(Km1 + 1);
      x(0) = 1.0;
      const T half_pi = T(0.5 * M_PI);
      for (size_type k = 1; k <= Km1; ++k) {
        T yk_1 = y(k-1) + half_pi;
        T sin_yk_1 = sin(yk_1);
        x(k) = x(k-1) * sin_yk_1; 
        x(k-1) *= cos(yk_1);
        if (k < Km1)
          lp += (Km1 - k) * log(fabs(sin_yk_1));
      }
      return x;
    }

    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    unit_vector_free(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      stan::error_handling::check_unit_vector("stan::prob::unit_vector_free", 
                                              "Unit vector variable", x);
      int Km1 = x.size() - 1;
      Matrix<T,Dynamic,1> y(Km1);
      T sumSq = x(Km1)*x(Km1);
      const T half_pi = T(M_PI/2.0);
      for (size_type k = Km1; --k >= 0; ) {
        y(k) = atan2(sqrt(sumSq),x(k)) - half_pi;
        sumSq += x(k)*x(k);
      }
      return y;
    }


    // SIMPLEX

    /**
     * Return the simplex corresponding to the specified free vector.  
     * A simplex is a vector containing values greater than or equal
     * to 0 that sum to 1.  A vector with (K-1) unconstrained values
     * will produce a simplex of size K.
     *
     * The transform is based on a centered stick-breaking process.
     *
     * @param y Free vector input of dimensionality K - 1.
     * @return Simplex of dimensionality K.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    simplex_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& y) {

      // cut & paste simplex_constrain(Eigen::Matrix,T) w/o Jacobian
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      using stan::math::inv_logit;
      using stan::math::logit;
      using stan::math::log1m;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;


      int Km1 = y.size();
      Matrix<T,Dynamic,1> x(Km1 + 1);
      T stick_len(1.0);
      for (size_type k = 0; k < Km1; ++k) {
        T z_k(inv_logit(y(k) - log(Km1 - k))); 
        x(k) = stick_len * z_k;
        stick_len -= x(k); 
      }
      x(Km1) = stick_len;
      return x;
    }

    /**
     * Return the simplex corresponding to the specified free vector
     * and increment the specified log probability reference with 
     * the log absolute Jacobian determinant of the transform. 
     *
     * The simplex transform is defined through a centered
     * stick-breaking process.
     * 
     * @param y Free vector input of dimensionality K - 1.
     * @param lp Log probability reference to increment.
     * @return Simplex of dimensionality K.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    simplex_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& y, 
                      T& lp) {

      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;
      using stan::math::inv_logit;
      using stan::math::logit;
      using stan::math::log1m;
      using stan::math::log1p_exp;

      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      int Km1 = y.size(); // K = Km1 + 1
      Matrix<T,Dynamic,1> x(Km1 + 1);
      T stick_len(1.0);
      for (size_type k = 0; k < Km1; ++k) {
        double eq_share = -log(Km1 - k); // = logit(1.0/(Km1 + 1 - k));
        T adj_y_k(y(k) + eq_share);
        T z_k(inv_logit(adj_y_k));
        x(k) = stick_len * z_k;
        lp += log(stick_len);
        lp -= log1p_exp(-adj_y_k);
        lp -= log1p_exp(adj_y_k);
        stick_len -= x(k); // equivalently *= (1 - z_k);
      }
      x(Km1) = stick_len; // no Jacobian contrib for last dim
      return x;
    }

    /**
     * Return an unconstrained vector that when transformed produces
     * the specified simplex.  It applies to a simplex of dimensionality
     * K and produces an unconstrained vector of dimensionality (K-1).
     *
     * <p>The simplex transform is defined through a centered
     * stick-breaking process.
     * 
     * @param x Simplex of dimensionality K.
     * @return Free vector of dimensionality (K-1) that transfroms to
     * the simplex.
     * @tparam T Type of scalar.
     * @throw std::domain_error if x is not a valid simplex
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    simplex_free(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) {
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;
      using stan::math::logit;

      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      stan::error_handling::check_simplex("stan::prob::simplex_free", "Simplex variable", x);
      int Km1 = x.size() - 1;
      Eigen::Matrix<T,Eigen::Dynamic,1> y(Km1);
      T stick_len(x(Km1));
      for (size_type k = Km1; --k >= 0; ) {
        stick_len += x(k);
        T z_k(x(k) / stick_len);
        y(k) = logit(z_k) + log(Km1 - k); 
        // note: log(Km1 - k) = logit(1.0 / (Km1 + 1 - k));
      }
      return y;
    }


    // ORDERED 
    
    /**
     * Return an increasing ordered vector derived from the specified
     * free vector.  The returned constrained vector will have the
     * same dimensionality as the specified free vector.
     *
     * @param x Free vector of scalars.
     * @return Positive, increasing ordered vector.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    ordered_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;

      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      size_type k = x.size();
      Matrix<T,Dynamic,1> y(k);
      if (k == 0)
        return y;
      y[0] = x[0];
      for (size_type i = 1; i < k; ++i)
        y[i] = y[i-1] + exp(x[i]);
      return y;
    }

    /**
     * Return a positive valued, increasing ordered vector derived
     * from the specified free vector and increment the specified log
     * probability reference with the log absolute Jacobian determinant
     * of the transform.  The returned constrained vector
     * will have the same dimensionality as the specified free vector.
     *
     * @param x Free vector of scalars.
     * @param lp Log probability reference.
     * @return Positive, increasing ordered vector. 
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    ordered_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, T& lp) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;

      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      for (size_type i = 1; i < x.size(); ++i)
        lp += x(i);
      return ordered_constrain(x);
    }



    /**
     * Return the vector of unconstrained scalars that transform to
     * the specified positive ordered vector.
     *
     * <p>This function inverts the constraining operation defined in 
     * <code>ordered_constrain(Matrix)</code>,
     *
     * @param y Vector of positive, ordered scalars.
     * @return Free vector that transforms into the input vector.
     * @tparam T Type of scalar.
     * @throw std::domain_error if y is not a vector of positive,
     *   ordered scalars.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    ordered_free(const Eigen::Matrix<T,Eigen::Dynamic,1>& y) {
      stan::error_handling::check_ordered("stan::prob::ordered_free", 
                                          "Ordered variable", y);
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      size_type k = y.size();
      Matrix<T,Dynamic,1> x(k);
      if (k == 0) 
        return x;
      x[0] = y[0];
      for (size_type i = 1; i < k; ++i)
        x[i] = log(y[i] - y[i-1]);
      return x;
    }
    

    // POSITIVE ORDERED 
    
    /**
     * Return an increasing positive ordered vector derived from the specified
     * free vector.  The returned constrained vector will have the
     * same dimensionality as the specified free vector.
     *
     * @param x Free vector of scalars.
     * @return Positive, increasing ordered vector.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    positive_ordered_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      size_type k = x.size();
      Matrix<T,Dynamic,1> y(k);
      if (k == 0)
        return y;
      y[0] = exp(x[0]);
      for (size_type i = 1; i < k; ++i)
        y[i] = y[i-1] + exp(x[i]);
      return y;
    }

    /**
     * Return a positive valued, increasing positive ordered vector derived
     * from the specified free vector and increment the specified log
     * probability reference with the log absolute Jacobian determinant
     * of the transform.  The returned constrained vector
     * will have the same dimensionality as the specified free vector.
     *
     * @param x Free vector of scalars.
     * @param lp Log probability reference.
     * @return Positive, increasing ordered vector. 
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    positive_ordered_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, T& lp) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      for (size_type i = 0; i < x.size(); ++i)
        lp += x(i);
      return positive_ordered_constrain(x);
    }



    /**
     * Return the vector of unconstrained scalars that transform to
     * the specified positive ordered vector.
     *
     * <p>This function inverts the constraining operation defined in 
     * <code>positive_ordered_constrain(Matrix)</code>,
     *
     * @param y Vector of positive, ordered scalars.
     * @return Free vector that transforms into the input vector.
     * @tparam T Type of scalar.
     * @throw std::domain_error if y is not a vector of positive,
     *   ordered scalars.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    positive_ordered_free(const Eigen::Matrix<T,Eigen::Dynamic,1>& y) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;

      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      stan::error_handling::check_positive_ordered("stan::prob::positive_ordered_free", 
                                                   "Positive ordered variable", 
                                                   y);

      size_type k = y.size();
      Matrix<T,Dynamic,1> x(k);
      if (k == 0) 
        return x;
      x[0] = log(y[0]);
      for (size_type i = 1; i < k; ++i)
        x[i] = log(y[i] - y[i-1]);
      return x;
    }
    

    // CHOLESKY FACTOR

    /**
     * Return the Cholesky factor of the specified size read from the
     * specified vector.  A total of (N choose 2) + N + (M - N) * N
     * elements are required to read an M by N Cholesky factor.
     * 
     * @tparam T Type of scalars in matrix
     * @param x Vector of unconstrained values
     * @param M Number of rows
     * @param N Number of columns
     * @return Cholesky factor
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    cholesky_factor_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x,
                              int M,
                              int N) {
      using std::exp;
      if (M < N)
        throw std::domain_error("cholesky_factor_constrain: num rows must be >= num cols");
      if (x.size() != ((N * (N + 1)) / 2 + (M - N) * N))
        throw std::domain_error("cholesky_factor_constrain: x.size() must"
                                " be (N * (N + 1)) / 2 + (M - N) * N");
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> y(M,N);
      T zero(0);
      int pos = 0;
      // upper square
      for (int m = 0; m < N; ++m) {
        for (int n = 0; n < m; ++n)
          y(m,n) = x(pos++);
        y(m,m) = exp(x(pos++));
        for (int n = m + 1; n < N; ++n)
          y(m,n) = zero;
      }
      // lower rectangle
      for (int m = N; m < M; ++m)
        for (int n = 0; n < N; ++n)
          y(m,n) = x(pos++);
      return y;
    }

    /**
     * Return the Cholesky factor of the specified size read from the
     * specified vector and increment the specified log probability
     * reference with the log Jacobian adjustment of the transform.  A
     * total of (N choose 2) + N + N * (M - N) free parameters are required to read
     * an M by N Cholesky factor.
     * 
     * @tparam T Type of scalars in matrix
     * @param x Vector of unconstrained values
     * @param M Number of rows
     * @param N Number of columns
     * @param lp Log probability that is incremented with the log Jacobian
     * @return Cholesky factor
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    cholesky_factor_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x,
                              int M,
                              int N,
                              T& lp) {
      // cut-and-paste from above, so checks twice

      using stan::math::sum;
      if (x.size() != ((N * (N + 1)) / 2 + (M - N) * N))
        throw std::domain_error("cholesky_factor_constrain: x.size() must be (k choose 2) + k");
      int pos = 0;
      std::vector<T> log_jacobians(N);
      for (int n = 0; n < N; ++n) {
        pos += n;
        log_jacobians[n] = x(pos++);
      }
      lp += sum(log_jacobians);  // optimized for autodiff vs. direct lp += 
      return cholesky_factor_constrain(x,M,N);
    }

    /**
     * Return the unconstrained vector of parameters correspdonding to
     * the specified Cholesky factor.  A Cholesky factor must be lower
     * triangular and have positive diagonal elements.
     *
     * @param y Cholesky factor.
     * @return Unconstrained parameters for Cholesky factor.
     * @throw std::domain_error If the matrix is not a Cholesky factor.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1>
    cholesky_factor_free(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y) {
      using std::log;
      if (!stan::error_handling::check_cholesky_factor("cholesky_factor_free", "y", y))
        throw std::domain_error("cholesky_factor_free: y is not a Cholesky factor");
      int M = y.rows();
      int N = y.cols();
      Eigen::Matrix<T,Eigen::Dynamic,1> x((N * (N + 1)) / 2 + (M - N) * N);
      int pos = 0;
      // lower triangle of upper square
      for (int m = 0; m < N; ++m) {
        for (int n = 0; n < m; ++n)
          x(pos++) = y(m,n);
        // diagonal of upper square
        x(pos++) = log(y(m,m));
      }
      // lower rectangle
      for (int m = N; m < M; ++m)
        for (int n = 0; n < N; ++n)
          x(pos++) = y(m,n);
      return x;
    }

    // CHOLESKY CORRELATION MATRIX

    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    cholesky_corr_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& y,
                            int K) {
      using std::sqrt;
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::square;
      int k_choose_2 = (K * (K - 1)) / 2;
      if (k_choose_2 != y.size()) {
        std::cout << "k_choose_2 = " << k_choose_2 << " y.size()=" << y.size() << std::endl;
        throw std::domain_error("y is not a valid unconstrained cholesky correlation matrix."
                                "Require (K choose 2) elements in y.");
      }
      Matrix<T,Dynamic,1> z(k_choose_2);
      for (int i = 0; i < k_choose_2; ++i)
        z(i) = corr_constrain(y(i));
      Matrix<T,Dynamic,Dynamic> x(K,K);
      if (K == 0) return x;
      T zero(0);
      for (int j = 1; j < K; ++j)
        for (int i = 0; i < j; ++i)
          x(i,j) = zero;
      x(0,0) = 1;
      int k = 0;
      for (int i = 1; i < K; ++i) {
        x(i,0) = z(k++);
        T sum_sqs(square(x(i,0)));
        for (int j = 1; j < i; ++j) {
          x(i,j) = z(k++) * sqrt(1.0 - sum_sqs);
          sum_sqs += square(x(i,j));
        }
        x(i,i) = sqrt(1.0 - sum_sqs);
      }
      return x;
    }

    // FIXME to match above after debugged
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    cholesky_corr_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& y,
                            int K,
                            T& lp) {
      using std::sqrt;
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::log1m;
      using stan::math::square;
      int k_choose_2 = (K * (K - 1)) / 2;
      if (k_choose_2 != y.size()) {
        std::cout << "k_choose_2 = " << k_choose_2 << " y.size()=" << y.size() << std::endl;
        throw std::domain_error("y is not a valid unconstrained cholesky correlation matrix."
                                " Require (K choose 2) elements in y.");
      }
      Matrix<T,Dynamic,1> z(k_choose_2);
      for (int i = 0; i < k_choose_2; ++i)
        z(i) = corr_constrain(y(i),lp);
      Matrix<T,Dynamic,Dynamic> x(K,K);
      if (K == 0) return x;
      T zero(0);
      for (int j = 1; j < K; ++j)
        for (int i = 0; i < j; ++i)
          x(i,j) = zero;
      x(0,0) = 1;
      int k = 0;
      for (int i = 1; i < K; ++i) {
        x(i,0) = z(k++);
        T sum_sqs = square(x(i,0));
        for (int j = 1; j < i; ++j) {
          lp += 0.5 * log1m(sum_sqs);
          x(i,j) = z(k++) * sqrt(1.0 - sum_sqs);
          sum_sqs += square(x(i,j));
        }
        x(i,i) = sqrt(1.0 - sum_sqs);
      }
      return x;
    }

    

    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1>
    cholesky_corr_free(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x) {
      using std::sqrt;
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::square;

      stan::error_handling::check_square("cholesky_corr_free", "x", x);
      // should validate lower-triangular, unit lengths

      int K = (x.rows() * (x.rows() - 1)) / 2;
      Matrix<T,Dynamic,1> z(K);
      int k = 0;
      for (int i = 1; i < x.rows(); ++i) {
        z(k++) = corr_free(x(i,0));
        double sum_sqs = square(x(i,0));
        for (int j = 1; j < i; ++j) {
          z(k++) = corr_free(x(i,j) / sqrt(1.0 - sum_sqs));
          sum_sqs += square(x(i,j));
        }
      }
      return z;
    }

    // CORRELATION MATRIX

    /**
     * Return the correlation matrix of the specified dimensionality
     * derived from the specified vector of unconstrained values.  The
     * input vector must be of length \f${k \choose 2} =
     * \frac{k(k-1)}{2}\f$.  The values in the input vector represent
     * unconstrained (partial) correlations among the dimensions.
     *
     * <p>The transform based on partial correlations is as specified
     * in
     *
     * <ul><li> Lewandowski, Daniel, Dorota Kurowicka, and Harry
     * Joe. 2009.  Generating random correlation matrices based on
     * vines and extended onion method.  <i>Journal of Multivariate
     * Analysis</i> <b>100</b>:1989–-2001.  </li></ul>
     *
     * <p>The free vector entries are first constrained to be
     * valid correlation values using <code>corr_constrain(T)</code>.
     * 
     * @param x Vector of unconstrained partial correlations.
     * @param k Dimensionality of returned correlation matrix.
     * @tparam T Type of scalar.
     * @throw std::invalid_argument if x is not a valid correlation
     * matrix.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
    corr_matrix_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x,
                  typename math::index_type<Eigen::Matrix<T,Eigen::Dynamic,1> >::type k) {

      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      size_type k_choose_2 = (k * (k - 1)) / 2;
      if (k_choose_2 != x.size())
        throw std::invalid_argument ("x is not a valid correlation matrix");
      Eigen::Array<T,Eigen::Dynamic,1> cpcs(k_choose_2);
      for (size_type i = 0; i < k_choose_2; ++i)
        cpcs[i] = corr_constrain(x[i]);
      return read_corr_matrix(cpcs,k); 
    }

    /**
     * Return the correlation matrix of the specified dimensionality
     * derived from the specified vector of unconstrained values.  The
     * input vector must be of length \f${k \choose 2} =
     * \frac{k(k-1)}{2}\f$.  The values in the input vector represent
     * unconstrained (partial) correlations among the dimensions.
     *
     * <p>The transform is as specified for
     * <code>corr_matrix_constrain(Matrix,size_t)</code>; the
     * paper it cites also defines the Jacobians for correlation inputs,
     * which are composed with the correlation constrained Jacobians 
     * defined in <code>corr_constrain(T,double)</code> for
     * this function.
     * 
     * @param x Vector of unconstrained partial correlations.
     * @param k Dimensionality of returned correlation matrix.
     * @param lp Log probability reference to increment.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
    corr_matrix_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
                  typename math::index_type<Eigen::Matrix<T,Eigen::Dynamic,1> >::type k,
                  T& lp) {
      using Eigen::Array;
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;
  
      size_type k_choose_2 = (k * (k - 1)) / 2;
      if (k_choose_2 != x.size())
        throw std::invalid_argument ("x is not a valid correlation matrix");
      Array<T,Dynamic,1> cpcs(k_choose_2);
      for (size_type i = 0; i < k_choose_2; ++i)
        cpcs[i] = corr_constrain(x[i],lp);
      return read_corr_matrix(cpcs,k,lp);
    }

    /**
     * Return the vector of unconstrained partial correlations that
     * define the specified correlation matrix when transformed.
     *
     * <p>The constraining transform is defined as for
     * <code>corr_matrix_constrain(Matrix,size_t)</code>.  The
     * inverse transform in this function is simpler in that it only
     * needs to compute the \f$k \choose 2\f$ partial correlations
     * and then free those.
     * 
     * @param y The correlation matrix to free.
     * @return Vector of unconstrained values that produce the
     * specified correlation matrix when transformed.
     * @tparam T Type of scalar.
     * @throw std::domain_error if the correlation matrix has no
     *    elements or is not a square matrix.
     * @throw std::runtime_error if the correlation matrix cannot be
     *    factorized by factor_cov_matrix() or if the sds returned by
     *    factor_cov_matrix() on log scale are unconstrained.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    corr_matrix_free(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y) {
      using Eigen::Array;
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      size_type k = y.rows();
      if (y.cols() != k)
        throw std::domain_error("y is not a square matrix");
      if (k == 0)
        throw std::domain_error("y has no elements");
      size_type k_choose_2 = (k * (k-1)) / 2;
      Array<T,Dynamic,1> x(k_choose_2);
      Array<T,Dynamic,1> sds(k);
      bool successful = factor_cov_matrix(y,x,sds);
      if (!successful)
        throw std::runtime_error("factor_cov_matrix failed on y");
      for (size_type i = 0; i < k; ++i) {
        // sds on log scale unconstrained
        if (fabs(sds[i] - 0.0) >= CONSTRAINT_TOLERANCE) {
          std::stringstream s;
          s << "all standard deviations must be zero."
            << " found log(sd[" << i << "])=" << sds[i] << std::endl;
          BOOST_THROW_EXCEPTION(std::runtime_error(s.str()));
        }
      }
      return x.matrix();
    }


    // COVARIANCE MATRIX

    /**
     * Return the symmetric, positive-definite matrix of dimensions K
     * by K resulting from transforming the specified finite vector of
     * size K plus (K choose 2).
     *
     * <p>See <code>cov_matrix_free()</code> for the inverse transform.
     *
     * @param x The vector to convert to a covariance matrix.
     * @param K The number of rows and columns of the resulting
     * covariance matrix.
     * @throws std::domain_error if (x.size() != K + (K choose 2)).
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
    cov_matrix_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
                 typename math::index_type<Eigen::Matrix<T,Eigen::Dynamic,1> >::type K) {
      using std::exp;

      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;
      using stan::math::multiply_lower_tri_self_transpose;
      typedef typename index_type<Matrix<T,Dynamic,Dynamic> >::type size_type;

      Matrix<T,Dynamic,Dynamic> L(K,K);
      if (x.size() != (K * (K + 1)) / 2) 
        throw std::domain_error("x.size() != K + (K choose 2)");
      int i = 0;
      for (size_type m = 0; m < K; ++m) {
        for (int n = 0; n < m; ++n)
          L(m,n) = x(i++);
        L(m,m) = exp(x(i++));
        for (size_type n = m + 1; n < K; ++n) 
          L(m,n) = 0.0;
      }
      return multiply_lower_tri_self_transpose(L); 
    }

    
    /**
     * Return the symmetric, positive-definite matrix of dimensions K
     * by K resulting from transforming the specified finite vector of
     * size K plus (K choose 2).
     *
     * <p>See <code>cov_matrix_free()</code> for the inverse transform.
     *
     * @param x The vector to convert to a covariance matrix.
     * @param K The dimensions of the resulting covariance matrix.
     * @param lp Reference
     * @throws std::domain_error if (x.size() != K + (K choose 2)).
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
    cov_matrix_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
         typename math::index_type<Eigen::Matrix<T,
                                                 Eigen::Dynamic,
                                                 Eigen::Dynamic> >::type K,
         T& lp) {
      using std::exp;

      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,Dynamic> >::type size_type;

      if (x.size() != (K * (K + 1)) / 2) 
        throw std::domain_error("x.size() != K + (K choose 2)");
      Matrix<T,Dynamic,Dynamic> L(K,K);
      int i = 0;
      for (size_type m = 0; m < K; ++m) {
        for (size_type n = 0; n < m; ++n)
          L(m,n) = x(i++);
        L(m,m) = exp(x(i++));
        for (size_type n = m + 1; n < K; ++n) 
          L(m,n) = 0.0;
      }
      // Jacobian for complete transform, including exp() above
      lp += (K * stan::math::LOG_2); // needless constant; want propto
      for (int k = 0; k < K; ++k)
        lp += (K - k + 1) * log(L(k,k)); // only +1 because index from 0
      return L * L.transpose();
      // return tri_multiply_transpose(L); 
    }

    /**
     * The covariance matrix derived from the symmetric view of the
     * lower-triangular view of the K by K specified matrix is freed
     * to return a vector of size K + (K choose 2).  
     *
     * This is the inverse of the <code>cov_matrix_constrain()</code>
     * function so that for any finite vector <code>x</code> of size K
     * + (K choose 2),
     *
     * <code>x == cov_matrix_free(cov_matrix_constrain(x,K))</code>.
     *
     * In order for this round-trip to work (and really for this
     * function to work), the symmetric view of its lower-triangular
     * view must be positive definite.
     * 
     * @param y Matrix of dimensions K by K such that he symmetric
     * view of the lower-triangular view is positive definite.
     * @return Vector of size K plus (K choose 2) in (-inf,inf)
     * that produces
     * @throw std::domain_error if <code>y</code> is not square, 
     * has zero dimensionality, or has a non-positive diagonal element.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    cov_matrix_free(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y) {
      using std::log;
      int K = y.rows();
      if (y.cols() != K)
        throw std::domain_error("y is not a square matrix");
      if (K == 0)
        throw std::domain_error("y has no elements");
      for (int k = 0; k < K; ++k)
        if (!(y(k,k) > 0.0)) 
          throw std::domain_error("y has non-positive diagonal");
      Eigen::Matrix<T,Eigen::Dynamic,1> x((K * (K + 1)) / 2);
      // FIXME: see Eigen LDLT for rank-revealing version -- use that
      // even if less efficient?
      Eigen::LLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > 
        llt(y.rows());
      llt.compute(y);
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L = llt.matrixL();
      int i = 0;
      for (int m = 0; m < K; ++m) {
        for (int n = 0; n < m; ++n)
          x(i++) = L(m,n);
        x(i++) = log(L(m,m));
      }
      return x;
    }

    /**
     * Return the covariance matrix of the specified dimensionality
     * derived from constraining the specified vector of unconstrained
     * values.  The input vector must be of length \f$k \choose 2 +
     * k\f$.  The first \f$k \choose 2\f$ values in the input
     * represent unconstrained (partial) correlations and the last
     * \f$k\f$ are unconstrained standard deviations of the dimensions.
     *
     * <p>The transform scales the correlation matrix transform defined
     * in <code>corr_matrix_constrain(Matrix,size_t)</code>
     * with the constrained deviations.  
     * 
     * @param x Input vector of unconstrained partial correlations and
     * standard deviations.
     * @param k Dimensionality of returned covariance matrix.
     * @return Covariance matrix derived from the unconstrained partial
     * correlations and deviations.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
    cov_matrix_constrain_lkj(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
                         size_t k) {
      size_t k_choose_2 = (k * (k - 1)) / 2;
      Eigen::Array<T,Eigen::Dynamic,1> cpcs(k_choose_2);
      int pos = 0;
      for (size_t i = 0; i < k_choose_2; ++i)
        cpcs[i] = corr_constrain(x[pos++]);
      Eigen::Array<T,Eigen::Dynamic,1> sds(k);
      for (size_t i = 0; i < k; ++i)
        sds[i] = positive_constrain(x[pos++]);
      return read_cov_matrix(cpcs, sds);
    }

    /**
     * Return the covariance matrix of the specified dimensionality
     * derived from constraining the specified vector of unconstrained
     * values and increment the specified log probability reference
     * with the log absolute Jacobian determinant.  
     *
     * <p>The transform is defined as for
     * <code>cov_matrix_constrain(Matrix,size_t)</code>.
     *
     * <p>The log absolute Jacobian determinant is derived by
     * composing the log absolute Jacobian determinant for the
     * underlying correlation matrix as defined in
     * <code>cov_matrix_constrain(Matrix,size_t,T&)</code> with
     * the Jacobian of the transfrom of the correlation matrix
     * into a covariance matrix by scaling by standard deviations.
     * 
     * @param x Input vector of unconstrained partial correlations and
     * standard deviations.
     * @param k Dimensionality of returned covariance matrix.
     * @param lp Log probability reference to increment.
     * @return Covariance matrix derived from the unconstrained partial
     * correlations and deviations.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
    cov_matrix_constrain_lkj(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
                         size_t k, 
                         T& lp) {
      size_t k_choose_2 = (k * (k - 1)) / 2;
      Eigen::Array<T,Eigen::Dynamic,1> cpcs(k_choose_2);
      int pos = 0;
      for (size_t i = 0; i < k_choose_2; ++i)
        cpcs[i] = corr_constrain(x[pos++], lp);
      Eigen::Array<T,Eigen::Dynamic,1> sds(k);
      for (size_t i = 0; i < k; ++i)
        sds[i] = positive_constrain(x[pos++],lp);
      return read_cov_matrix(cpcs, sds, lp);
    }

    /**
     * Return the vector of unconstrained partial correlations and
     * deviations that transform to the specified covariance matrix.
     *
     * <p>The constraining transform is defined as for
     * <code>cov_matrix_constrain(Matrix,size_t)</code>.  The
     * inverse first factors out the deviations, then applies the
     * freeing transfrom of <code>corr_matrix_free(Matrix&)</code>.
     *
     * @param y Covariance matrix to free.
     * @return Vector of unconstrained values that transforms to the
     * specified covariance matrix.
     * @tparam T Type of scalar.
     * @throw std::domain_error if the correlation matrix has no
     *    elements or is not a square matrix.
     * @throw std::runtime_error if the correlation matrix cannot be
     *    factorized by factor_cov_matrix()
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    cov_matrix_free_lkj(
            const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y) {

      using Eigen::Array;
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,Dynamic> >::type size_type;

      size_type k = y.rows();
      if (y.cols() != k)
        throw std::domain_error("y is not a square matrix");
      if (k == 0)
        throw std::domain_error("y has no elements");
      size_type k_choose_2 = (k * (k-1)) / 2;
      Array<T,Dynamic,1> cpcs(k_choose_2);
      Array<T,Dynamic,1> sds(k);
      bool successful = factor_cov_matrix(y,cpcs,sds);
      if (!successful)
        throw std::runtime_error ("factor_cov_matrix failed on y");
      Matrix<T,Dynamic,1> x(k_choose_2 + k);
      size_type pos = 0;
      for (size_type i = 0; i < k_choose_2; ++i)
        x[pos++] = cpcs[i];
      for (size_type i = 0; i < k; ++i)
        x[pos++] = sds[i];
      return x;
    }

  }

}

#endif
