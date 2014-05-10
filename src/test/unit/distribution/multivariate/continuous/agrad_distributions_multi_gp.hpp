class agrad_distributions_multi_gp : public ::testing::Test {
protected:
  virtual void SetUp() {
    y.resize(3,2);
    y << 2.0, -2.0, 11.0,
         0.0, 1.0, 5.0;
    y2.resize(3,2);
    y2 << 15.0, 1.0, -5.0,
          2.0, 3.0, -10.0;

    w.resize(3,1);
    w << 1.0, 1.0, 3.0;
    w2.resize(3,1);
    w2 << 6.0, 2.0, 6.0;

    Sigma.resize(2,2);
    Sigma << 9.0, -3.0,
            -3.0,  4.0;
    Sigma2.resize(2,2);
    Sigma2 << 3.0, 1.0,
              1.0, 5.0;
  }
  
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y2;
  Eigen::Matrix<double,Eigen::Dynamic,1> w;
  Eigen::Matrix<double,Eigen::Dynamic,1> w2;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Sigma;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Sigma2;
};
