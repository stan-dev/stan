#include <gtest/gtest.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <boost/math/distributions/students_t.hpp>
#include <stdio.h>

#include <stan/mcmc/chains.hpp>

class Models_BasicDistributions_Binormal : public ::testing::Test {
protected:
  virtual void SetUp() {
    FILE *in;
    if(!(in = popen("make path_separator --no-print-directory", "r")))
      throw std::runtime_error("\"make path_separator\" has failed.");
    path_separator += fgetc(in);
    pclose(in);
    
    model.append("models").append(path_separator);
    model.append("basic_distributions").append(path_separator);
    model.append("binormal");

    output1 = model + "1.csv";
    output2 = model + "2.csv";
    
    expected_y1 = 0.0;
    expected_y2 = 0.0;
  }
  std::string path_separator;
  std::string model;
  std::string output1;
  std::string output2;
  
  double expected_y1;
  double expected_y2;
};

TEST_F(Models_BasicDistributions_Binormal,RunModel) {
  std::string command;
  command = model;
  command += " --samples=";
  command += output1;
  EXPECT_EQ(0, system(command.c_str())) 
    << "Can not execute command: " << command << std::endl;
            
  
  command = model;
  command += " --samples=";
  command += output2;
  EXPECT_EQ(0, system(command.c_str()))
    << "Can not execute command: " << command << std::endl;
}
TEST_F(Models_BasicDistributions_Binormal, y1) {
  std::vector<std::string> names;
  std::vector<std::vector<size_t> > dimss;
  stan::mcmc::read_variables(output1, 2,
                             names, dimss);

  stan::mcmc::chains<> c(2, names, dimss);
  stan::mcmc::add_chain(c, 0U, output1, 2U);
  stan::mcmc::add_chain(c, 1U, output2, 2U);

  size_t index;
  std::vector<size_t> idxs;
  idxs.push_back(0);
  index = c.get_total_param_index(c.param_name_to_index("y"), 
                                  idxs);

  double neff = c.effective_sample_size(index);

  boost::math::students_t t_dist(neff-1.0);  
  double T = boost::math::quantile(t_dist, 0.975);
  
  EXPECT_NEAR(expected_y1, c.mean(index), T*sqrt(c.variance(index)/neff));
}
TEST_F(Models_BasicDistributions_Binormal, y2) {
  std::vector<std::string> names;
  std::vector<std::vector<size_t> > dimss;
  stan::mcmc::read_variables(output1, 2,
                             names, dimss);

  stan::mcmc::chains<> c(2, names, dimss);
  stan::mcmc::add_chain(c, 0U, output1, 2U);
  stan::mcmc::add_chain(c, 1U, output2, 2U);

  size_t index;
  std::vector<size_t> idxs;
  idxs.push_back(1);
  index = c.get_total_param_index(c.param_name_to_index("y"), 
                                  idxs);

  double neff = c.effective_sample_size(index);

  boost::math::students_t t_dist(neff-1.0);  
  double T = boost::math::quantile(t_dist, 0.975);
  
  EXPECT_NEAR(expected_y1, c.mean(index), T*sqrt(c.variance(index)/neff));
}
