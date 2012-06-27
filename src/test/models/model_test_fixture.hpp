#ifndef __TEST__MODELS__MODEL_TEST_FIXTURE_HPP__
#define __TEST__MODELS__MODEL_TEST_FIXTURE_HPP__

#include <gtest/gtest.h>
#include <test/models/utility.hpp>
#include <stan/mcmc/chains.hpp>
#include <utility>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/binomial.hpp>

/** 
 * Model_Test_Fixture is a test fixture for google test
 * to aid in running models.
 *
 * Derived classes must define:
 *   static std::vector<std::string> get_model_path()
 *   static bool has_data()
 *   - indicates whether the model has data
 */
template <class Derived>
class Model_Test_Fixture : public ::testing::Test {
  
public:
  static char path_separator;
  static std::string model_path;
  static stan::mcmc::chains<> *chains;
  static size_t num_chains;

  /** 
   * SetUpTestCase() called by google test once
   * a test case.
   *
   * Sets:
   *  - path_separator: '\' for Windows, '/' for all else.
   *  - base_name:      location of model without ".stan"
   *  - chains:         creates a chains object by reading
   *                    the header of a csv file
   */
  static void SetUpTestCase() {
    path_separator = get_path_separator();
    model_path = convert_model_path(get_model_path());
      
    chains = create_chains();
  }
    
  /** 
   * TearDownTestCase() called by google test once
   * a test case.
   *
   * Deletes chains. 
   */
  static void TearDownTestCase() {
    delete chains;
  }

  /** 
   * Returns the path as a string with the appropriate
   * path separator.
   * 
   * @param model_path vector of strings representing path to the model
   * 
   * @return the string representation of the path with the appropriate
   *    path separator.
   */
  static std::string
  convert_model_path(const std::vector<std::string> model_path) {
    std::string path;
    if (model_path.size() > 0) {
      path.append(model_path[0]);
      for (size_t i = 1; i < model_path.size(); i++) {
        path.append(1, path_separator);
        path.append(model_path[i]);
      }
    }
    return path;
  }

  /** 
   * Returns the csv file for the chain number.
   * 
   * @param chain the chain number
   * 
   * @return the file location of the csv file
   */
  static std::string get_csv_file(size_t chain) {
    std::stringstream csv_file;
    csv_file << model_path << "." << chain << ".csv";
    return csv_file.str();
  }

  /** 
   * Returns the command to run in a shell on the OS.
   * 
   * @param chain the chain number 
   * 
   * @return a string command that can be run from a shell.
   */
  static std::string get_command(size_t chain) {
    std::stringstream command;
    command << model_path;
    command << " --samples=" << get_csv_file(chain);
    if (has_data()) {
      command << " --data=" << model_path << ".Rdata";
    }
    command << " --refresh=2000";
    return command.str();
  }


  /** 
   * Populates the chains object with data from csv files.
   */
  static void populate_chains() {
    if (chains->num_kept_samples() == 0U) {
      for (size_t chain = 0U; chain < num_chains; chain++) {
        stan::mcmc::add_chain(*chains, chain, get_csv_file(chain), 2U);
      }
    }
  }
  
    
  /** 
   * Runs the model num_chains times.
   * Populates the chains object after running the model.
   */
  static void run_model() {
    for (size_t chain = 0; chain < num_chains; chain++) {
      std::string command = get_command(chain);
      EXPECT_EQ(0, system(command.c_str()))
        << "Can not execute command: " << command << std::endl;
    }
    populate_chains();
  }
    
  /** 
   * Creates a chains object.
   *
   * Runs the model for 0 iterations to read the
   * names and dimensions of the parameters.
   * 
   * @return An initialized chains object.
   */
  static stan::mcmc::chains<>* create_chains() {
    std::string command = get_command(0U);
    command += " --iter=0";
    EXPECT_EQ(0, system(command.c_str()))
      << "Can not build header using: " << command << std::endl;
      
    std::vector<std::string> names;
    std::vector<std::vector<size_t> > dimss;
    stan::mcmc::read_variables(get_csv_file(0U), 2,
                               names, dimss);
      
    return (new stan::mcmc::chains<>(num_chains, names, dimss));
  }


  /** 
   * Return the path to the model (without the extension) as
   * a vector.
   * 
   * @return the path to the model
   */
  static std::vector<std::string> get_model_path() {
    return Derived::get_model_path();
  }
  
  /**
   * Return true if the model has data.
   *
   * @return true if the model has data;
   *         false otherwise.
   */
  static bool has_data() {
    return Derived::has_data();
  }
  
  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    return Derived::get_expected_values();
  }

};
  
template<class Derived> 
char Model_Test_Fixture<Derived>::path_separator;

template<class Derived> 
stan::mcmc::chains<> *Model_Test_Fixture<Derived>::chains;

template<class Derived>
size_t Model_Test_Fixture<Derived>::num_chains = 2;

template<class Derived>
std::string Model_Test_Fixture<Derived>::model_path;



TYPED_TEST_CASE_P(Model_Test_Fixture);

TYPED_TEST_P(Model_Test_Fixture, RunModel) {
  TypeParam::run_model();
}

TYPED_TEST_P(Model_Test_Fixture, ChainsTest) {
  stan::mcmc::chains<> *c = TypeParam::chains;
  size_t num_chains = c->num_chains();
  size_t num_params = c->num_params();
  for (size_t chain = 0; chain < num_chains; chain++) {
    for (size_t param = 0; param < num_params; param++) {
      EXPECT_TRUE(c->variance(chain, param) > 0)
        << "Chain " << chain << ", param " << param
        << ": variance is 0";
    }
  }
}


TYPED_TEST_P(Model_Test_Fixture, ExpectedValuesTest) {
  using std::vector;
  using std::pair;
  using std::sqrt;
  using std::abs;
  using std::setw;

  using boost::math::students_t;
  using boost::math::binomial;
  using boost::math::quantile;
  
  vector<pair<size_t, double> > expected_values = TypeParam::get_expected_values();
  size_t n = expected_values.size();
  if (n == 0)
    return;

  stan::mcmc::chains<> *c = TypeParam::chains;
  double alpha = 0.05;
  if (n < 3)
    alpha = 0.01;

  
  int failed = 0;
  std::stringstream err_message;
  for (size_t i = 0; i < n; i++) {
    size_t index = expected_values[i].first;
    double expected_mean = expected_values[i].second;

    double neff = c->effective_sample_size(index);
    double sample_mean = c->mean(index);
    double se = c->sd(index) / sqrt(neff);
    double z = quantile(students_t(neff-1.0), 1 - alpha/2.0);

    if (abs(expected_mean - sample_mean) > z*se) {
      failed++;
      // want the error message to have which, what, how
      err_message << "parameter index: " << index
                  << "\n\texpected:    " << setw(10) << expected_mean
                  << "\n\tsampled:     " << setw(10) << sample_mean
                  << "\n\tneff:        " << setw(10) << neff
                  << "\n\tsplit R.hat: " << setw(10) << c->split_potential_scale_reduction(index)
                  << "\n\tz:           " << setw(10) << z
                  << "\n\tse:          " << setw(10) << se
                  << "\n\n\tabs(diff) > z * se: " 
                  << abs(expected_mean - sample_mean) << " > " << z*se << "\n\n";
    }
  }
  
  if (failed == 0)
    return;
  

  double p = 1 - cdf(binomial(n, alpha), failed);
  // this test should fail less than 0.1% of the time.
  if (p < 0.001) {
    EXPECT_EQ(0, failed)
      << "Failed " << failed << " of " << expected_values.size() << " comparisons\n"
      << "p: " << p << std::endl
      << "------------------------------------------------------------\n"
      << err_message.str();
  }
}

REGISTER_TYPED_TEST_CASE_P(Model_Test_Fixture,
                           RunModel,
                           ChainsTest,
                           ExpectedValuesTest);

#endif
