#include "kalman_filter.h"

// Student code start
#include <iostream>
#include <math.h>
using namespace std;
// Student code end

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {
    //cout<<"Kalman Filter constructor running...\n"<<endl;
}

KalmanFilter::~KalmanFilter() {}

// void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        // MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  // x_ = x_in;
  // P_ = P_in;
  // F_ = F_in;
  // H_ = H_in;
  // R_ = R_in;
  // Q_ = Q_in;
// }

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

// For Lidar Measurements
void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  // State Update using measurements from Lidar
  // First step is Kalman Gain Calculation
  // Size of z_pred is (2 by 1)
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // Sencond step is kalman filter state update using Kalman Gain.
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  // State Covariance Update
  P_ = (I - K * H_) * P_;
}

// For Radar Measurements
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  // State Update using measurements from Radar
  // Size of z_pred is 3 by 1
  VectorXd z_pred(3);
  // x_(O) is px, x_(1) is py, x_(2) is vx, x_(3) is vy
  double rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
  double phi = atan2(x_(1), x_(0)); 
  double rho_dot;
  // Note here we are testing if the  radial distance is greater than zero, to avoid divison by zero
  // Once certain, rho_dot is assigned (px*vx + py*vy)/(px^2+py^2)
  if(fabs(rho) > 0.0001){
    rho_dot = (x_(0)*x_(2) + x_(1)*x_(3))/rho;
  }
  else{
    rho_dot = 0; 
  }

  z_pred << rho, 
            phi, 
            rho_dot;
  // Note here we have &z as argument of UpdateEKF function, thus it is a pointer to measurement_pack.raw_measurements_
  // Note z is coming from measurements, while z_pred is coming from Hx' or h(x)
  // Size of z and z_pred is (3 by 1)
  VectorXd y = z - z_pred;
  // y has residual, rho, phi and rho_dot.
  // We want residual phi to be in between -pi to pi, so normalizing it.
  y(1) = atan2(sin(y(1)), cos(y(1)));
  
  // First step now is Kalman Gain Calculation
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // Sencond step is kalman filter state update using Kalman Gain.
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_; 
}
