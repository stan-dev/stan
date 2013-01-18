#ifndef __TEST__AGRAD__DISTRIBUTIONS__TEST_FIXTURE_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__TEST_FIXTURE_HPP___

#include <stdexcept>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix.hpp>
#include <stan/agrad/agrad.hpp>
#include <test/agrad/distributions/utility.hpp>

using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;
using stan::agrad::var;
using stan::scalar_type;
using stan::is_vector;
using stan::is_constant;
using stan::is_constant_struct;



/** 
 * To test a distribution, define a subclass of AgradDistributionTest.
 * Implement each of the functions.
 * 
 */
class AgradDistributionTest {
public:
  
  virtual void valid_values(vector<vector<double> >& /*parameters*/,
			    vector<double>& /* log_prob */) {
    throw std::runtime_error("valid_values() not implemented");
  }
  
  // don't need to list nan. checked by the test.
  virtual void invalid_values(vector<size_t>& /*index*/, 
                              vector<double>& /*value*/) {
    throw std::runtime_error("valid_values() not implemented");
  }

  // also include:
  /*
    template <bool propto, 
	    typename T0, typename T1, typename T2,
	    typename T3, typename T4, typename T5, 
	    typename T6, typename T7, typename T8, 
	    typename T9>
  typename return_type<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type 
  log_prob(const T0&, const T1&, const T2&,
	   const T3&, const T4&, const T5&, 
	   const T6&, const T7&, const T8&, const T9&) {
  }

  template <bool propto, 
	    typename T0, typename T1, typename T2,
	    typename T3, typename T4, typename T5, 
	    typename T6, typename T7, typename T8, 
	    typename T9, 
	    class Policy>
  typename return_type<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type 
  log_prob(const T0&, const T1&, const T2&,
	   const T3&, const T4&, const T5&, 
	   const T6&, const T7&, const T8&, const T9&, const Policy&) {
  }

  template <typename T0, typename T1, typename T2,
	    typename T3, typename T4, typename T5, 
	    typename T6, typename T7, typename T8, 
	    typename T9>
  var log_prob_function(const T0&, const T1&, const T2&,
			const T3&, const T4&, const T5&, 
			const T6&, const T7&, const T8&, const T9&, const Policy&) {
  }
  */
};




using boost::mpl::at_c;
template<class T>
class AgradDistributionTestFixture : public ::testing::Test {
public:
  typename at_c<T,0>::type TestClass;
  typedef typename at_c<typename at_c<T,1>::type, 0>::type T0;
  typedef typename at_c<typename at_c<T,1>::type, 1>::type T1;
  typedef typename at_c<typename at_c<T,1>::type, 2>::type T2;
  typedef typename at_c<typename at_c<T,1>::type, 3>::type T3;
  typedef typename at_c<typename at_c<T,1>::type, 4>::type T4;
  typedef typename at_c<typename at_c<T,1>::type, 5>::type T5;
  typedef typename at_c<typename at_c<T,1>::type, 6>::type T6;
  typedef typename at_c<typename at_c<T,1>::type, 7>::type T7;
  typedef typename at_c<typename at_c<T,1>::type, 8>::type T8;
  typedef typename at_c<typename at_c<T,1>::type, 9>::type T9;
  
  void call_all_versions() {
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
    T0 p0 = get_params<T0>(parameters, 0);
    T1 p1 = get_params<T1>(parameters, 1);
    T2 p2 = get_params<T2>(parameters, 2);
    T3 p3 = get_params<T3>(parameters, 3);
    T4 p4 = get_params<T4>(parameters, 4);
    T5 p5 = get_params<T5>(parameters, 5);
    T6 p6 = get_params<T6>(parameters, 6);
    T7 p7 = get_params<T7>(parameters, 7);
    T8 p8 = get_params<T8>(parameters, 8);
    T9 p9 = get_params<T9>(parameters, 9);
    
    EXPECT_NO_THROW(({ TestClass.template log_prob
	    <T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
	    (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9); }))
      << "Calling log_prob throws exception with default parameters";

    EXPECT_NO_THROW(({ TestClass.template log_prob
	    <true, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
	    (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9); }))
      << "Calling log_prob throws exception with propto=true";

    EXPECT_NO_THROW(({ TestClass.template log_prob
	    <false, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
	    (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9); }))
      << "Calling log_prob throws exception with propto=false";
    
    EXPECT_NO_THROW(({ TestClass.template log_prob
	    <true, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, errno_policy>
	    (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9); }))
      << "Calling log_prob throws exception with propto=true, errno_policy";

    EXPECT_NO_THROW(({ TestClass.template log_prob
	    <false, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, errno_policy>
	    (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9); }))
      << "Calling log_prob throws exception with propto=false, errno_policy";
  }

  void test_valid_values() {
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      T0 p0 = get_params<T0>(parameters, n, 0);
      T1 p1 = get_params<T1>(parameters, n, 1);
      T2 p2 = get_params<T2>(parameters, n, 2);
      T3 p3 = get_params<T3>(parameters, n, 3);
      T4 p4 = get_params<T4>(parameters, n, 4);
      T5 p5 = get_params<T5>(parameters, n, 5);
      T6 p6 = get_params<T6>(parameters, n, 6);
      T7 p7 = get_params<T7>(parameters, n, 7);
      T8 p8 = get_params<T8>(parameters, n, 8);
      T9 p9 = get_params<T9>(parameters, n, 9);

      EXPECT_NO_THROW(({ TestClass.template log_prob
	      <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
	      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }))
	<< "Valid parameters failed at index: " << n << " -- " 
	<< parameters[n];
    }
  }

  vector<double> first_valid_params() {
    vector<vector<double> > params;
    vector<double> log_prob;

    TestClass.valid_values(params, log_prob); 
    return params[0];
  }
  /*
  double e() {
    return 1e-8;
    }*/
};
TYPED_TEST_CASE_P(AgradDistributionTestFixture);

TYPED_TEST_P(AgradDistributionTestFixture, CallAllVersions) {
  this->call_all_versions();
}

TYPED_TEST_P(AgradDistributionTestFixture, ValidValues) {
  this->test_valid_values();
}

TYPED_TEST_P(AgradDistributionTestFixture, InvalidValues) {
  FAIL() << "not implemented";
}

TYPED_TEST_P(AgradDistributionTestFixture, Propto) {
  FAIL() << "not implemented";
}

TYPED_TEST_P(AgradDistributionTestFixture, FiniteDiff) {
  FAIL() << "not implemented";
}

TYPED_TEST_P(AgradDistributionTestFixture, Function) {
}

TYPED_TEST_P(AgradDistributionTestFixture, RepeatAsVector) {
  FAIL() << "not implemented";
}

TYPED_TEST_P(AgradDistributionTestFixture, Vectorized) {
  FAIL() << "not implemented";
}

REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   CallAllVersions,
                           ValidValues,
			   InvalidValues,
			   Propto,
			   FiniteDiff,
			   Function,
			   RepeatAsVector,
			   Vectorized);




class AgradCdfTest {
};

template<class T>
class AgradCdfTestFixture : public ::testing::Test {
};

TYPED_TEST_CASE_P(AgradCdfTestFixture);


TYPED_TEST_P(AgradCdfTestFixture, DoesBlah) {
  FAIL() << "not implemented";
}

REGISTER_TYPED_TEST_CASE_P(AgradCdfTestFixture,
                           DoesBlah);


#endif
