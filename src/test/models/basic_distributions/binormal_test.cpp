#include <stan/mcmc/mcmc_output.hpp>
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstdio>


class binormal : public ::testing::Test {
protected:
  virtual void SetUp() {
    model = "models/basic_distributions/binormal";
    output1 = model + "1.csv";
    output2 = model + "2.csv";
    factory.addFile(output1);
    factory.addFile(output2);
  }
  std::string model;
  std::string output1;
  std::string output2;
    
  stan::mcmc::mcmc_output_factory factory;
};

TEST_F(binormal,runModel) {
  std::string command;
  command = model;
  command += " --samples=";
  command += output1;
  system(command.c_str());
  
  command = model;
  command += " --samples=";
  command += output2;
  system(command.c_str());
}

TEST_F(binormal, y1) {
  stan::mcmc::mcmc_output y1 = factory.create("y.1");
  double neff = y1.effectiveSize();
  double rHat = y1.splitRHat();
  
  //EXPECT_NEAR(0, ); FIXME: need mean, variance
  EXPECT_NEAR(1, rHat, 0.01);
}

TEST_F(binormal, y2) {
  stan::mcmc::mcmc_output y1 = factory.create("y.2");
}

