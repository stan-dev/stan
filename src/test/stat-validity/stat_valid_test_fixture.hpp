#ifndef __TEST__STAT_VALIDITY__STAT_VALID_TEST_FIXTURE_HPP__
#define __TEST__STAT_VALIDITY__STAT_VALID_TEST_FIXTURE_HPP__

#include <gtest/gtest.h>
#include <test/CmdStan/models/utility.hpp>
#include <stan/mcmc/chains.hpp>
#include <utility>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/binomial.hpp>

#include <fstream>
#include <string>

// Derived classes must define:
//   static std::vector<std::string> get_model_path()
//  static bool has_data()

template <class Derived>
class Stat_Valid_Test_Fixture : public ::testing::Test {
  
public:
  static char path_separator;
  static std::string model_path;
  static stan::mcmc::chains<> *chains;
  static std::vector<std::string> command_outputs;
  static int num_chains;
  static const int num_sampler_vars;

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
    model_path = convert_model_path(get_model_path());
    chains = create_chains();
  }
  
  // TearDownTestCase() called by google test once a test case.
  static void TearDownTestCase() { delete chains; }

  // Returns the csv file for the chain number.
  static std::string get_csv_file(int chain) {
    std::stringstream csv_file;
    csv_file << model_path << "." << chain << ".csv";
    return csv_file.str();
  }

  static std::string get_command(int chain) {
    std::stringstream command;
    command << model_path;
    command << " id=" << chain;
    if (has_data()) command << " data file=" << model_path << ".data.R";
    if (has_init()) command << " init=" << model_path << ".init.R";
    command << " output file=" << get_csv_file(chain)
            << " refresh=" << num_samples();
    return command.str();
  }

  static void populate_chains() {
    if (chains->num_kept_samples() == 0U) {
      for (int chain = 1U; chain <= num_chains; chain++) {
        std::ifstream ifstream(get_csv_file(chain).c_str());
        chains->add(stan::io::stan_csv_reader::parse(ifstream));
        ifstream.close();
      }
    }
  }
  
  static void run_model() {
    for (int chain = 1; chain <= num_chains; chain++) {

      std::string command = get_command(chain);
      
      std::stringstream method;
      method << " sample"
             << " num_samples=" << num_samples()
             << " num_warmup=" << num_warmup();
      
      command += method.str();
      
      run_command_output out;
      out = run_command(command);
      command_outputs.push_back(out.output);
      
    }
    
    populate_chains();
    
  }
  
  // Runs the model for 0 iterations to read the
  // names and dimensions of the parameters.
  static stan::mcmc::chains<>* create_chains() {
    
    std::string command = get_command(1U);
    command += " sample num_samples=0 num_warmup=0";
    
    run_command_output out = run_command(command);
    EXPECT_FALSE(out.hasError)
      << "Can not build header using: " << out;
      
    std::ifstream ifstream;
    ifstream.open(get_csv_file(1).c_str());
    stan::io::stan_csv stan_csv = stan::io::stan_csv_reader::parse(ifstream);
    ifstream.close();
    
    return (new stan::mcmc::chains<>(stan_csv));
  }

  static std::vector<std::string> get_model_path() {
    return Derived::get_model_path();
  }
  
  static bool has_data()   { return Derived::has_data();    }
  static bool has_init()   { return Derived::has_init();    }
  static int num_warmup()  { return Derived::num_warmup();  }
  static int num_samples() { return Derived::num_samples(); }
  
  static std::vector<std::pair<std::string, double> >
  get_expected_values() { return Derived::get_expected_values(); }

};
  
template<class Derived> 
stan::mcmc::chains<> *Stat_Valid_Test_Fixture<Derived>::chains;

template<class Derived>
int Stat_Valid_Test_Fixture<Derived>::num_chains = 50;

template<class Derived>
std::string Stat_Valid_Test_Fixture<Derived>::model_path;

template<class Derived>
std::vector<std::string> Stat_Valid_Test_Fixture<Derived>::command_outputs;

template<class Derived>
const int Stat_Valid_Test_Fixture<Derived>::num_sampler_vars = 6;

TYPED_TEST_CASE_P(Stat_Valid_Test_Fixture);

TYPED_TEST_P(Stat_Valid_Test_Fixture, RunModel) {
  TypeParam::run_model();
}

TYPED_TEST_P(Stat_Valid_Test_Fixture, ConvergenceTest) {
  
  stan::mcmc::chains<> *c = TypeParam::chains;
  int num_params = c->num_params();

  for (int param = TypeParam::num_sampler_vars; param < num_params; param++) {
    EXPECT_LT(c->split_potential_scale_reduction(param), 1.1)
      << "Param " << param << ", " << c->param_name(param)
      << ": Split R_{hat} > 1.1" << std::endl;
  }
}

TYPED_TEST_P(Stat_Valid_Test_Fixture, ExpectedValuesExistenceTest) {
  
  
  stan::mcmc::chains<> *c = TypeParam::chains;
  
  // Check expected values
  std::vector<std::pair<std::string, double> > expected_values = TypeParam::get_expected_values();
  int n = expected_values.size();
  if (n == 0) return;
  
  for (int i = 0; i < n; i++) {
    std::string name = expected_values[i].first;
    EXPECT_GT(c->index(name), 0)
      << name << " does not exist in the compiled model!\n";
  }
  
}

TYPED_TEST_P(Stat_Valid_Test_Fixture, ExpectedValuesTest) {
  
  using std::vector;
  using std::pair;
  using std::sqrt;
  using std::fabs;
  using std::setw;

  using boost::math::students_t;
  using boost::math::binomial;
  using boost::math::quantile;
  
  stan::mcmc::chains<> *c = TypeParam::chains;
  
  // Hypotehesis test thresholds
  double p_t_low = 0.1;
  double p_t_mid = 0.5;
  double p_t_high = 1 - p_t_low;
  
  double p_binomial = 0.999;
  double p_bin_low = 0.5 * (1 - p_binomial);
  double p_bin_high = 1 - p_bin_low;
  
  binomial bin_low(c->num_chains(), p_t_low);
  double n_bin_low_low = quantile(bin_low, p_bin_low);
  double n_bin_low_high = quantile(bin_low, p_bin_high);
  
  binomial bin_mid(c->num_chains(), p_t_mid);
  double n_bin_mid_low = quantile(bin_mid, p_bin_low);
  double n_bin_mid_high = quantile(bin_mid, p_bin_high);
  
  binomial bin_high(c->num_chains(), p_t_high);
  double n_bin_high_low = quantile(bin_high, p_bin_low);
  double n_bin_high_high = quantile(bin_high, p_bin_high);
  
  // Check expected values
  vector<pair<std::string, double> > expected_values = TypeParam::get_expected_values();
  int n = expected_values.size();
  if (n == 0) return;

  for (int i = 0; i < n; i++) {
    
    std::string name = expected_values[i].first;
    double expected_mean = expected_values[i].second;
    
    if (c->index(name) < 0) continue;
    
    int n_fail_low = 0;
    int n_fail_mid = 0;
    int n_fail_high = 0;
    
    for (int j = 0; j < c->num_chains(); ++j) {
    
      double n_eff = c->effective_sample_size(j, name);
      double sample_mean = c->mean(j, name);
      double se = c->sd(j, name) / sqrt(n_eff);
      
      double z = (sample_mean - expected_mean) / se;
      
      boost::math::students_t t_dist(n_eff - 1);
      double low_threshold = - quantile(complement(t_dist, p_t_low));
      double mid_threshold = - quantile(complement(t_dist, p_t_mid));
      double high_threshold = - quantile(complement(t_dist, p_t_high));

      
      if (z < low_threshold) ++n_fail_low;
      if (z < mid_threshold) ++n_fail_mid;
      if (z < high_threshold) ++n_fail_high;
      
    }
  
    EXPECT_EQ(false, (n_fail_low <= n_bin_low_low) || (n_fail_low > n_bin_low_high))
      << "Failed low-tail ensemble test for parameter " << name << "\n";
    
    EXPECT_EQ(false, (n_fail_mid <= n_bin_mid_low) || (n_fail_mid > n_bin_mid_high))
      << "Failed median ensemble test for parameter " << name << "\n";
    
    EXPECT_EQ(false, (n_fail_high <= n_bin_high_low) || (n_fail_high > n_bin_high_high))
      << "Failed high-tail ensemble test for parameter " << name << "\n";
    
  }

}

REGISTER_TYPED_TEST_CASE_P(Stat_Valid_Test_Fixture,
                           RunModel,
                           ConvergenceTest,
                           ExpectedValuesExistenceTest,
                           ExpectedValuesTest);

#endif
