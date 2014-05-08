class agrad_distributions_multi_normal_multi_row : public ::testing::Test {
protected:
  virtual void SetUp() {
    y.resize(1,3);
    y << 2.0, -2.0, 11.0;
    y2.resize(1,3);
    y2 << 15.0, 1.0, -5.0;

    mu.resize(3,1);
    mu << 1.0, -1.0, 3.0;
    mu2.resize(3,1);
    mu2 << 6.0, 2.0, -6.0;

    Sigma.resize(3,3);
    Sigma << 9.0, -3.0, 0.0,
      -3.0,  4.0, 0.0,
      0.0, 0.0, 5.0;
    Sigma2.resize(3,3);
    Sigma2 << 3.0, 1.0, 0.0,
      1.0,  5.0, -2.0,
      0.0, -2.0, 9.0;
  }
  
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y2;
  Eigen::Matrix<double,Eigen::Dynamic,1> mu;
  Eigen::Matrix<double,Eigen::Dynamic,1> mu2;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Sigma;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Sigma2;
};
