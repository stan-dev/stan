#ifndef __STAN__PROB__TRANSFORM_HPP__
#define __STAN__PROB__TRANSFORM_HPP__

#include <cstddef>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <boost/multi_array.hpp>
#include <boost/throw_exception.hpp>
#include <stan/math/matrix.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/special_functions.hpp>

namespace stan {
  
  namespace prob {

    using Eigen::Array;
    using Eigen::Dynamic;
    using Eigen::LDLT;
    using Eigen::Matrix;
    using Eigen::Array;
    using Eigen::DiagonalMatrix;

    const double CONSTRAINT_TOLERANCE = 1E-8;


    /**
     * This function is intended to make starting values, given a covariance matrix Sigma
     * The transformations are hard coded as log for standard deviations and Fisher
     * transformations (atanh()) of CPCs
     * @author Ben Goodrich
     * @return false if any of the diagonals of Sigma are 0
     */
    template<typename T>
    bool
    factor_cov_matrix(Array<T,Dynamic,1>& CPCs, // will fill this unbounded
                      Array<T,Dynamic,1>& sds,  // will fill this unbounded
                      const Matrix<T,Dynamic,Dynamic>& Sigma) {

      size_t K = sds.rows();

      sds = Sigma.diagonal().array();
      if( (sds <= 0.0).any() ) return false;
      sds = sds.sqrt();
  
      DiagonalMatrix<T,Dynamic> D(K);
      D.diagonal() = sds.inverse();
      sds = sds.log(); // now unbounded
  
      Matrix<T,Dynamic,Dynamic> R = D * Sigma * D;
      R.diagonal().setOnes(); // to hopefully prevent pivoting due to floating point error
      LDLT<Matrix<T,Dynamic,Dynamic> > ldlt;
      ldlt = R.ldlt();
      if( !ldlt.isPositive() ) return false;
      Matrix<T,Dynamic,Dynamic> U = ldlt.matrixU();

      size_t position = 0;
      size_t pull = K - 1;

      Array<T,Dynamic,1> temp = U.row(0).tail(pull);
      CPCs.head(pull) = temp;
  
      Array<T,Dynamic,1> acc(K);
      acc(0) = -0.0;
      acc.tail(pull) = 1.0 - temp.square();
      for(size_t i = 1; i < (K - 1); i++) {
        position += pull;
        pull--;
        temp = U.row(i).tail(pull).array();
        temp /= sqrt(acc.tail(pull) / acc(i));
        CPCs.segment(position, pull) = temp;
        acc.tail(pull) *= 1.0 - temp.square();
      }
      CPCs = 0.5 * ( (1.0 + CPCs) / (1.0 - CPCs) ).log(); // now unbounded
      return true;
    }

    // MATRIX TRANSFORMS +/- JACOBIANS

    /**
     * Return the Cholesky factor of the correlation matrix of the specified
     * dimensionality corresponding to the specified canonical partial correlations.
     * 
     * It is generally better to work with the Cholesky factor rather than the
     * correlation matrix itself when the determinant, inverse, etc. of the
     * correlation matrix is needed for some statistical calculation.
     *
     * <p>See <code>read_corr_matrix(Array,size_t,T)</code>
     * for more information.
     *
     * @param CPCs The (K choose 2) canonical partial correlations in (-1,1).
     * @param K Dimensionality of correlation matrix.
     * @return Cholesky factor of correlation matrix for specified canonical partial correlations.
     * @tparam T Type of underlying scalar.  
     * @author Ben Goodrich
     */
    template <typename T>
    Matrix<T,Dynamic,Dynamic>
    read_corr_L(const Array<T,Dynamic,1>& CPCs, // on (-1,1)
                const size_t K) {
      Array<T,Dynamic,1> temp;         // temporary holder
      Array<T,Dynamic,1> acc(K-1);     // accumlator of products
      acc.setOnes();
      Array<T,Dynamic,Dynamic> L(K,K); // Cholesky factor of correlation matrix
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
     * @return Cholesky factor of correlation matrix for specified canonical partial correlations.
     * @tparam T Type of underlying scalar.  
     * @author Ben Goodrich
     */
    template <typename T>
    Matrix<T,Dynamic,Dynamic>
    read_corr_matrix(const Array<T,Dynamic,1>& CPCs, // on (-1,1)
                     const size_t K) {
      Matrix<T,Dynamic,Dynamic> L = read_corr_L(CPCs, K);
      return L.template triangularView<Eigen::Lower>() * L.matrix().transpose();
    }
    
    /**
     * Return the Cholesky factor of the correlation matrix of the specified
     * dimensionality corresponding to the specified canonical partial correlations,
     * incrementing the specified scalar reference with the log
     * absolute determinant of the Jacobian of the transformation.
     *
     * <p>The implementation is Ben Goodrich's Cholesky
     * factor-based approach to the C-vine method of:
     * 
     * <ul><li>
     * Daniel Lewandowski, Dorota Kurowicka, and Harry Joe, 
     * Generating random correlation matrices based on vines and extended onion method
     * Journal of Multivariate Analysis 100 (2009) 1989–2001
     * </li></ul>
     *
     * // FIXME: explain which CPCs we're dealing with
     * 
     * @param CPCs The (K choose 2) canonical partial correlations in (-1,1).
     * @param K Dimensionality of correlation matrix.
     * @param log_prob Reference to variable to increment with the log
     * Jacobian determinant.
     * @return Cholesky factor of correlation matrix for specified partial correlations.
     * @tparam T Type of underlying scalar.  
     * @author Ben Goodrich
     */
    template <typename T>
    Matrix<T,Dynamic,Dynamic>
    read_corr_L(const Array<T,Dynamic,1>& CPCs, // on (-1,1)
                const size_t K,
                T& log_prob) {

      size_t k = 0; 
      size_t i = 0;
      T log_1cpc2;
      double lead = K - 2.0; 
      // no need to abs() because this Jacobian determinant 
      // is strictly positive (and triangular)
      // skip last row (odd indexing) because it adds nothing by design
      for (typename Array<T,Dynamic,1>::size_type j = 0; j < (CPCs.rows() - 1); ++j) {
        using stan::math::log1m;
        using stan::math::square;
        log_1cpc2 = log1m(square(CPCs[j]));
        log_prob += lead / 2.0 * log_1cpc2; // derivative of correlation wrt CPC
        i++;
        if (i > K) {
          k++;
          i = k + 1;
          lead = K - k - 1.0;
        }
      }
      return read_corr_L(CPCs, K);
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
     * @param CPCs The (K choose 2) canonical partial correlations in (-1,1).
     * @param K Dimensionality of correlation matrix.
     * @param log_prob Reference to variable to increment with the log
     * Jacobian determinant.
     * @return Correlation matrix for specified partial correlations.
     * @tparam T Type of underlying scalar.  
     * @author Ben Goodrich
     */
    template <typename T>
    Matrix<T,Dynamic,Dynamic>
    read_corr_matrix(const Array<T,Dynamic,1>& CPCs, // on (-1,1)
                     const size_t K,
                     T& log_prob) {

      Matrix<T,Dynamic,Dynamic> L = read_corr_L(CPCs, K, log_prob);
      return L.template triangularView<Eigen::Lower>() * L.matrix().transpose();
    }
    
    /** this is the function that should be called prior to evaluating the
     * density of any elliptical distribution
     * @return Cholesky factor of covariance matrix for specified partial correlations.
     * @author Ben Goodrich
     */
    template <typename T>
    Matrix<T,Dynamic,Dynamic>
    read_cov_L(const Array<T,Dynamic,1>& CPCs, // on (-1,1)
               const Array<T,Dynamic,1>& sds,  // on (0,inf)
               T& log_prob) {
      size_t K = sds.rows();
      // adjust due to transformation from correlations to covariances
      log_prob += (sds.log().sum() + log(2.0)) * K;
      return sds.matrix().asDiagonal() * read_corr_L(CPCs, K, log_prob);
    }

    /** a generally worse alternative to call prior to evaluating the density
     * of an elliptical distribution
     * @return Covariance matrix for specified partial correlations.
     * @author Ben Goodrich
     */
    template <typename T>
    Matrix<T,Dynamic,Dynamic>
    read_cov_matrix(const Array<T,Dynamic,1>& CPCs, // on (-1,1)
                    const Array<T,Dynamic,1>& sds,  // on (0,inf)
                    T& log_prob) {

      Matrix<T,Dynamic,Dynamic> L = read_cov_L(CPCs, sds, log_prob);
      return L.template triangularView<Eigen::Lower>() * L.matrix().transpose();
    }

    /** 
     *
     * Builds a covariance matrix from CPCs and standard deviations
     * @author Ben Goodrich
     */
    template<typename T>
    Matrix<T,Dynamic,Dynamic>
    read_cov_matrix(const Array<T,Dynamic,1>& CPCs,    // on (-1,1)
                    const Array<T,Dynamic,1>& sds) {   // on (0,inf)

      size_t K = sds.rows();
      DiagonalMatrix<T,Dynamic> D(K);
      D.diagonal() = sds;
      Matrix<T,Dynamic,Dynamic> L = D * read_corr_L(CPCs, K);
      return L.template triangularView<Eigen::Lower>() * L.matrix().transpose();
    }


    /** 
     * This function calculates the degrees of freedom for the t distribution
     * that corresponds to the shape parameter in the Lewandowski et. al. distribution
     * @author Ben Goodrich
     */
    template<typename T>
    const Array<T,Dynamic,1>
    make_nu(const T eta,             // hyperparameter on (0,inf), eta = 1 <-> correlation matrix is uniform
            const size_t K) {  // number of variables in covariance matrix
  
      Array<T,Dynamic,1> nu(K * (K - 1) / 2);
  
      T alpha = eta + (K - 2.0) / 2.0; // from Lewandowski et. al.
      // Best (1978) implies nu = 2 * alpha for the dof in a t 
      // distribution that generates a beta variate on (-1,1)
      T alpha2 = 2.0 * alpha; 

      for (size_t j = 0; j < (K - 1); j++) {
        nu(j) = alpha2;
      }
      size_t counter = K - 1;
      for (size_t i = 1; i < (K - 1); i++) {
        alpha -= 0.5;
        alpha2 = 2.0 * alpha;
        for (size_t j = i + 1; j < K; j++) {
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
     * @param lp Reference to log probability.
     * @return Transformed input.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline 
    T identity_constrain(const T x, T& lp) {
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
     * <p>\f$\log | \frac{d}{dx} \mbox{exp}(x) | = \log | \mbox{exp}(x) | =  x\f$.
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
     * <p>The input is validated using <code>stan::math::check_positive()</code>.
     * 
     * @param y Input scalar.
     * @return Unconstrained value that produces the input when constrained.
     * @tparam T Type of scalar.
     * @throw std::domain_error if the variable is negative.
     */
    template <typename T>
    T positive_free(const T y) {
      stan::math::check_positive("stan::prob::positive_free(%1%)", y, "y");
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
     * @param x Unconstrained scalar input.
     * @param lb Lower-bound on constrained ouptut.
     * @return Lower-bound constrained value correspdonding to inputs.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T lb_constrain(const T x, const double lb) {
      return exp(x) + lb;
    }

    /**
     * Return the lower-bounded value for the speicifed unconstrained input
     * and specified lower bound, incrementing the specified reference
     * with the log absolute Jacobian determinant of the transform.
     *
     * @param x Unconstrained scalar input.
     * @param lb Lower-bound on output.
     * @param lp Reference to log probability to increment.
     * @return Loer-bound constrained value corresponding to inputs.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T lb_constrain(const T x, const double lb, T& lp) {
      lp += x;
      return exp(x) + lb;
    }

    /**
     * Return the unconstrained value that produces the specified
     * lower-bound constrained value.
     * 
     * @param y Input scalar.
     * @param lb Lower bound.
     * @return Unconstrained value that produces the input when
     * constrained.
     * @tparam T Type of scalar.
     * @throw std::domain_error if y is lower than the lower bound.
     */
    template <typename T>
    inline
    T lb_free(const T y, const double lb) {
      stan::math::check_greater_or_equal("stan::prob::lb_free(%1%)",
                                         y, lb, "y");
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
     * @param x Free scalar.
     * @param ub Upper bound.
     * @return Transformed scalar with specified upper bound.
     * @tparam T Type of scalar.
     */
    template <typename T>
    T ub_constrain(const T x, const double ub) {
      return ub - exp(x);
    }

    /**
     * Return the upper-bounded value for the specified unconstrained
     * scalar and upper bound and increment the specified log probability
     * reference with the log absolute Jacobian determinant of the transform.
     *
     * <p>The transform is as specified for <code>ub_constrain(T,double)</code>.
     * The log absolute Jacobian determinant is
     *
     * <p>\f$ \log | \frac{d}{dx} -\mbox{exp}(x) + U | = \log | -\mbox{exp}(x) + 0 | = x\f$.
     *
     * @param x Free scalar.
     * @param ub Upper bound.
     * @param lp Log probability reference.
     * @return Transformed scalar with specified upper bound.
     * @tparam T Type of scalar.
     */
    template <typename T>
    T ub_constrain(const T x, const double ub, T& lp) {
      lp -= x;
      return ub - exp(x);
    }

    /**
     * Return the free scalar that corresponds to the specified
     * upper-bounded value with respect to the specified upper bound.
     *
     * <p>The transform is the reverse of the <code>ub_constrain(T,double)</code>
     * transform, 
     *
     * <p>\f$f^{-1}(y) = \log -(y - U)\f$
     *
     * <p>where \f$U\f$ is the upper bound.
     *
     * @param y Upper-bounded scalar.
     * @param ub Upper bound.
     * @return Free scalar corresponding to upper-bounded scalar.
     * @tparam T Type of scalar.
     * @throw std::invalid_argument if y is greater than the upper bound.
     */
    template <typename T>
    T ub_free(const T y, const double ub) {
      stan::math::check_less_or_equal("stan::prob::ub_free(%1%)",
                                      y, ub, "y");
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
     * @param x Free scalar to transform.
     * @param lb Lower bound.
     * @param ub Upper bound.
     * @return Lower- and upper-bounded scalar derived from transforming
     * the free scalar.
     * 
     * @tparam T Type of scalar.
     */
    template <typename T>
    T lub_constrain(const T x, double lb, double ub) {
      using stan::math::inv_logit;
      return lb + (ub - lb) * inv_logit(x);
    }

    /**
     * Return the lower- and upper-bounded scalar derived by
     * transforming the specified free scalar given the specified
     * lower and upper bounds and increment the specified log
     * probability with the log absolute Jacobian determinant.
     *
     * <p>The transform is as defined in <code>lub_constrain(T,double,double)</code>.
     * The log absolute Jacobian determinant is given by
     * 
     * <p>\f$\log \left| \frac{d}{dx} \left( L + (U-L) \mbox{logit}^{-1}(x) \right) \right|\f$
     * <p>\f$ {} = \log | (U-L) \, (\mbox{logit}^{-1}(x)) \, (1 - \mbox{logit}^{-1}(x)) |\f$
     * <p>\f$ {} = \log (U - L) + \log (\mbox{logit}^{-1}(x)) + \log (1 - \mbox{logit}^{-1}(x))\f$
     *
     * @param x Free scalar to transform.
     * @param lb Lower bound.
     * @param ub Upper bound.
     * @param lp Log probability scalar reference.
     * @return Lower- and upper-bounded scalar derived from transforming
     * the free scalar.
     * @tparam T Type of scalar.
     */
    template <typename T>
    T lub_constrain(const T x, const double lb, const double ub, T& lp) {

      T inv_logit_x;
      if (x > 0) {
        T exp_minus_x = exp(-x);
        inv_logit_x = 1.0 / (1.0 + exp_minus_x);
        lp += log(ub - lb) - x - 2 * log1p(exp_minus_x);
        // Prevent x from reaching one unless it really really should.
        if ((x < std::numeric_limits<double>::infinity()) && (inv_logit_x==1))
            inv_logit_x = 1 - 1e-15;
      } else {
        T exp_x = exp(x);
        inv_logit_x = 1.0 - 1.0 / (1.0 + exp_x);
        lp += log(ub - lb) + x - 2 * log1p(exp_x);
        // Prevent x from reaching zero unless it really really should.
        if ((x > -std::numeric_limits<double>::infinity()) && (inv_logit_x==0))
            inv_logit_x = 1e-100;
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
     * @param y Scalar input.
     * @param lb Lower bound.
     * @param ub Upper bound.
     * @return The free scalar that transforms to the input scalar
     * given the bounds.
     *
     * @tparam T Type of scalar.
     * @throw std::invalid_argument if the lower bound is greater than the upper bound,
     *   y is less than the lower bound, or
     *   y is greater than the upper bound
     */
    template <typename T>
    T lub_free(const T y, double lb, double ub) {
      using stan::math::logit;
      stan::math::check_bounded("stan::prob::lub_free(%1%)",
                                y, lb, ub, "y");
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
    T prob_free(const T y) {
      using stan::math::logit;
      stan::math::check_bounded("stan::prob::prob_free(%1%)",
                                y, 0, 1, "y, a probability,");
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
    T corr_constrain(const T x) {
      return tanh(x);
    }

    /**
     * Return the result of transforming the specified scalar to have
     * a valid correlation value between -1 and 1 (inclusive).
     *
     * <p>The transform used is as specified for <code>corr_constrain(T)</code>.
     * The log absolute Jacobian determinant is
     *
     * <p>\f$\log | \frac{d}{dx} \tanh x  | = \log (1 - \tanh^2 x)\f$.
     * 
     * @tparam T Type of scalar.
     */
    template <typename T>
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
     * <code>corr_constrain(T)</code>, which is the inverse hyperbolic tangent,
     *
     * <p>\f$f^{-1}(y) = \mbox{atanh}\, y = \frac{1}{2} \log \frac{y + 1}{y - 1}\f$.
     *
     * @param y Correlation scalar input.
     * @return Free scalar that transforms to the specified input.
     * @tparam T Type of scalar.
     */
    template <typename T>
    T corr_free(const T y) {
      stan::math::check_bounded("stan::prob::lub_free(%1%)",
                                y, -1, 1, "y, a correlation,");
      return atanh(y);
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
    Matrix<T,Dynamic,1> simplex_constrain(const Matrix<T,Dynamic,1>& y) {
      // cut and paste from  simplex_constrain(Matrix,T) w/o Jacobian
      using stan::math::logit;
      using stan::math::inv_logit;
      using stan::math::log1m;
      int Km1 = y.size();
      Matrix<T,Dynamic,1> x(Km1 + 1);
      T stick_len(1.0);
      for (int k = 0; k < Km1; ++k) {
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
     * The simplex transform is defined through a centered stick-breaking
     * process.
     * 
     * @param y Free vector input of dimensionality K - 1.
     * @param lp Log probability reference to increment.
     * @return Simplex of dimensionality K.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Matrix<T,Dynamic,1> simplex_constrain(const Matrix<T,Dynamic,1>& y, 
                                          T& lp) {
      using stan::math::logit;
      using stan::math::inv_logit;
      using stan::math::log1p_exp;
      using stan::math::log1m;
      int Km1 = y.size(); // K = Km1 + 1
      Matrix<T,Dynamic,1> x(Km1 + 1);
      T stick_len(1.0);
      for (int k = 0; k < Km1; ++k) {
        double eq_share = -log(Km1 - k); // = logit(1.0/(Km1 + 1 - k));
        T adj_y_k(y(k) + eq_share);
        T z_k(inv_logit(adj_y_k));
        x(k) = stick_len * z_k;
        lp += log(stick_len);
        lp -= log1p_exp(-adj_y_k);
        lp -= log1p_exp(adj_y_k);
        // lp += log(x(k)) + log1m(z_k);
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
     * <p>The simplex transform is defined through a centered stick-breaking
     * process.
     * 
     * @param x Simplex of dimensionality K.
     * @return Free vector of dimensionality (K-1) that transfroms to
     * the simplex.
     * @tparam T Type of scalar.
     * @throw std::domain_error if x is not a valid simplex
     */
    template <typename T>
    Matrix<T,Dynamic,1> simplex_free(const Matrix<T,Dynamic,1>& x) {
      using stan::math::logit;
      stan::math::check_simplex("stan::prob::simplex_free(%1%)", x, "x");
      int Km1 = x.size() - 1;
      Matrix<T,Dynamic,1> y(Km1);
      T stick_len(x(Km1));
      for (int k = Km1; --k >= 0; ) {
        stick_len += x(k);
        T z_k(x(k) / stick_len);
        y(k) = logit(z_k) + log(Km1 - k); 
        // log(Km-k) = logit(1.0 / (Km1 + 1 - k));
      }
      return y;
    }



    // POSITIVE ORDERED 
    
    /**
     * Return a positive valued, increasing ordered vector derived
     * from the specified free vector.  The returned constrained vector
     * will have the same dimensionality as the specified free vector.
     *
     * <p>The transform is defined using sums of exponentiations, where for
     * each dimension \f$k\f$,
     *
     * <p>\f$ f(x)[k] = \sum_{k' = 0}^{k} \exp(x[k])\f$
     *  
     * @param x Free vector of scalars.
     * @return Positive, increasing ordered vector.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Matrix<T,Dynamic,1> pos_ordered_constrain(const Matrix<T,Dynamic,1>& x) {
      typename Matrix<T,Dynamic,1>::size_type k = x.size();
      Matrix<T,Dynamic,1> y(k);
      if (k > 0)
        y[0] = exp(x[0]);
      for (typename Matrix<T,Dynamic,1>::size_type i = 1; i < k; ++i)
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
     * <p>The transform is defined as for <code>pos_ordered_constrain(Matrix<T,Dynamic,1>)</code>.
     * The log absolute Jacobian determinant reduces neatly because the Jacobian
     * is lower triangular, 
     *
     * <p>\f$\log \left| J \right| \f$
     *
     * <p>\f$= \log \left| \begin{array}{c} \nabla f(x)[0] \\ \vdots \\ \nabla f(x)[K-1] \end{array}\right|\f$
     * 
     * <p>\f${} = \log \left| \begin{array}{cccc}
     * \exp(x[0]) & 0 & \cdots & 0
     * \\ \exp(x[0]) & \exp(x[1]) & \cdots & 0
     * \\ \vdots & \vdots & \vdots & \vdots 
     * \\ \exp(x[0]) & \exp(x[1]) & \cdots & \exp(x[K-1])
     * \end{array} \right|\f$
     * 
     * <p>\f${} = \log \prod_{k=0}^{K-1} \exp(x[k])\f$
     *
     * <p>\f${} = \sum_{k=0}^{K-1} x[k]\f$.
     *  
     * @param x Free vector of scalars.
     * @param lp Log probability reference.
     * @return Positive, increasing ordered vector. 
     * @tparam T Type of scalar.
     */
    template <typename T>
    Matrix<T,Dynamic,1> pos_ordered_constrain(const Matrix<T,Dynamic,1>& x, T& lp) {
      lp += x.sum();
      return pos_ordered_constrain(x);
    }



    /**
     * Return the vector of unconstrained scalars that transform to
     * the specified positive ordered vector.
     *
     * <p>This function inverts the constraining operation defined in 
     * <code>pos_ordered_constrain(Matrix)</code>,
     *
     * <p>\f$f^{-1}(y)[k] = \log y[k] - \sum_{k' = 0}^{k-1} \log y[k]\f$
     *
     * @param y Vector of positive, ordered scalars.
     * @return Free vector that transforms into the input vector.
     * @tparam T Type of scalar.
     * @throw std::domain_error if y is not a vector of positive,
     *   ordered scalars.
     */
    template <typename T>
    Matrix<T,Dynamic,1> pos_ordered_free(const Matrix<T,Dynamic,1>& y) {
      stan::math::check_pos_ordered("stan::prob::pos_ordered_free(%1%)", y, "y");
      size_t k = y.size();
      Matrix<T,Dynamic,1> x(k);
      if (k == 0) 
        return x;
      x[0] = log(y[0]);
      for (size_t i = 1; i < k; ++i)
        x[i] = log(y[i] - y[i-1]);
      return x;
    }
    

    // CORRELATION MATRIX
    /**
     * Return the correlation matrix of the specified dimensionality
     * derived from the specified vector of unconstrained values.  The
     * input vector must be of length \f${k \choose 2} =
     * \frac{k(k-1)}{2}\f$.  The values in the input vector represent
     * unconstrained (partial) correlations among the dimensions.
     *
     * <p>The transform based on partial correlations is as specified in 
     *
     * <ul><li>
     * Lewandowski, Daniel, Dorota Kurowicka, and Harry Joe. 2009.
     * Generating random correlation matrices based on vines and extended onion method.
     * <i>Journal of Multivariate Analysis</i> <b>100</b>:1989–-2001.
     * </li></ul>
     *
     * <p>The free vector entries are first constrained to be
     * valid correlation values using <code>corr_constrain(T)</code>.
     * 
     * @param x Vector of unconstrained partial correlations.
     * @param k Dimensionality of returned correlation matrix.
     * @tparam T Type of scalar.
     * @throw std::invalid_argument if x is not a valid correlation matrix.
     */
    template <typename T>
    Matrix<T,Dynamic,Dynamic> corr_matrix_constrain(const Matrix<T,Dynamic,1>& x,
                                                    typename Matrix<T,Dynamic,1>::size_type k) {
      typedef typename Matrix<T,Dynamic,1>::size_type size_type;
      size_type k_choose_2 = (k * (k - 1)) / 2;
      if (k_choose_2 != x.size())
        throw std::invalid_argument ("x is not a valid correlation matrix");
      Array<T,Dynamic,1> cpcs(k_choose_2);
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
    Matrix<T,Dynamic,Dynamic> corr_matrix_constrain(const Matrix<T,Dynamic,1>& x, 
                                                    typename Matrix<T,Dynamic,1>::size_type k,
                                                    T& lp) {
      typedef typename Matrix<T,Dynamic,1>::size_type size_type;
      size_type k_choose_2 = (k * (k - 1)) / 2;
      if (k_choose_2 != x.size())
        throw std::invalid_argument ("x is not a valid correlation matrix");
      
      Array<T,Dynamic,1> cpcs(k_choose_2);
      for (size_type i = 0; i < k_choose_2; ++i)
        cpcs[i] = corr_constrain(x[i],lp);
      return read_corr_matrix(cpcs,k,lp);
    }

    /**
     * Return the vector of unconstrained partial correlations that define the
     * specified correlation matrix when transformed.  
     *
     * <p>The constraining transform is defined as for
     * <code>corr_matrix_constrain(Matrix,size_t)</code>.  The
     * inverse transform in this function is simpler in that it only
     * needs to compute the \f$k \choose 2\f$ partial correlations
     * and then free those.
     * 
     * @param y The correlation matrix to free.
     * @return Vector of unconstrained values that produce the specified
     * correlation matrix when transformed.
     * @tparam T Type of scalar.
     * @throw std::domain_error if the correlation matrix has no elements or
     *    is not a square matrix.
     * @throw std::runtime_error if the correlation matrix cannot be factorized
     *    by factor_cov_matrix() or if the sds returned by factor_cov_matrix()
     *    on log scale are unconstrained.
     */
    template <typename T>
    Matrix<T,Dynamic,1> corr_matrix_free(const Matrix<T,Dynamic,Dynamic>& y) {
      typedef typename Matrix<T,Dynamic,Dynamic>::size_type size_type;
      size_type k = y.rows();
      if (y.cols() != k || k == 0)
        throw std::domain_error("y is not a square matrix or there are no elements");
      size_type k_choose_2 = (k * (k-1)) / 2;
      Array<T,Dynamic,1> x(k_choose_2);
      Array<T,Dynamic,1> sds(k);
      bool successful = factor_cov_matrix(x,sds,y);
      if (!successful)
        throw std::runtime_error ("y cannot be factorized by factor_cov_matrix");
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
    Matrix<T,Dynamic,Dynamic> cov_matrix_constrain(const Matrix<T,Dynamic,1>& x, 
                                                   size_t k) {
      size_t k_choose_2 = (k * (k - 1)) / 2;
      Array<T,Dynamic,1> cpcs(k_choose_2);
      int pos = 0;
      for (size_t i = 0; i < k_choose_2; ++i)
        cpcs[i] = corr_constrain(x[pos++]);
      Array<T,Dynamic,1> sds(k);
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
     * <p>The transform is defined as for <code>cov_matrix_constrain(Matrix,size_t)</code>.
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
    Matrix<T,Dynamic,Dynamic> cov_matrix_constrain(const Matrix<T,Dynamic,1>& x, 
                                                   size_t k, 
                                                   T& lp) {
      size_t k_choose_2 = (k * (k - 1)) / 2;
      Array<T,Dynamic,1> cpcs(k_choose_2);
      int pos = 0;
      for (size_t i = 0; i < k_choose_2; ++i)
        cpcs[i] = corr_constrain(x[pos++], lp);
      Array<T,Dynamic,1> sds(k);
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
     * @throw std::domain_error if the correlation matrix has no elements or
     *    is not a square matrix.
     * @throw std::runtime_error if the correlation matrix cannot be factorized
     *    by factor_cov_matrix()
     */
    template <typename T>
    Matrix<T,Dynamic,1> cov_matrix_free(const Matrix<T,Dynamic,Dynamic>& y) {
      typedef typename Matrix<T,Dynamic,Dynamic>::size_type size_type;
      size_type k = y.rows();
      if (y.cols() != k || k == 0)
        throw std::domain_error("y is not a square matrix or there are no elements");
      size_type k_choose_2 = (k * (k-1)) / 2;
      Array<T,Dynamic,1> cpcs(k_choose_2);
      Array<T,Dynamic,1> sds(k);
      bool successful = factor_cov_matrix(cpcs,sds,y);
      if (!successful)
        throw std::runtime_error ("y cannot be factorized by factor_cov_matrix");
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
