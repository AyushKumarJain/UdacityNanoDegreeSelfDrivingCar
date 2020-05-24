#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

// Student Code Starts
// Newly added <math.h> and namespace std
#include <math.h>
using namespace std;
// Student Code Ends

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4); // Jacobian Matrix

  //measurement covariance matrix - laser (2 by 2 as we have only px and py measured)
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar (3 by 3 as we have only rho, phi and rho_dot measured)
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
  // Student Code Starts
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.x_ = VectorXd(4);
  ekf_.Q_ = MatrixXd(4, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  ekf_.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;
  // // Covariance Initialised 
  // ekf_.P_ << 1, 0, 0, 0,
  //           0, 1, 0, 0,
  //           0, 0, 1000, 0,
  //           0, 0, 0, 1000;
  // set the noise components
  noise_ax = 9;
  noise_ay = 9;    
  // Student Code Ends
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates 
      //         and initialize state.
      // Getting measurement of radial distance, direction and velocity from measurement_pack 
      const double rho = measurement_pack.raw_measurements_(0);
      const double phi = measurement_pack.raw_measurements_(1);
      const double rho_dot = measurement_pack.raw_measurements_(2);
      // Creating x vector of Extended Kalman Filter. 
      ekf_.x_ << rho*cos(phi), // x position
                 rho*sin(phi), // y position 
                 rho_dot*cos(phi), // x direction velocity
                 rho_dot*sin(phi); // y direction velocity
      // Covariance Initialised 
      ekf_.P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // TODO: Initialize state.
      // Lidar will be giving distance to the object
      const double px = measurement_pack.raw_measurements_(0);
      const double py = measurement_pack.raw_measurements_(1)
      ekf_.x_ << px, 
                 py, 
                 0, 
                 0;
      // Covariance Initialised 
      ekf_.P_ << 1, 0, 0, 0,
                 0, 1, 0, 0,
              0, 0, 1000, 0,
              0, 0, 0, 1000;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    // Student Code Start
    previous_timestamp_ = measurement_pack.timestamp_;
    // Student Code End
    return;
  }

  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  // Calculate the time elapsed in seconds.
  const float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  // Store the current timestamp in previous timestamp, so that dt can be calculated for the next round.
  previous_timestamp_ = measurement_pack.timestamp_;
  // Calculate t^2, t^3, t^4 to be used to find Covariance Matrix Q 
  const float dt_2 = dt * dt;
  const float dt_3 = dt_2 * dt;
  const float dt_4 = dt_3 * dt;

  // Modify the F matrix so that the time is integrated
  ekf_.F_ << 1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1;
  // ekf_.F_(0, 2) = dt;
  // ekf_.F_(1, 3) = dt;
  // Update Process Covariance Matrix Q after dt time step 
  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
              0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
              dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
              0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // TODO: Radar updates
    // Calculate Jacobian to update non linear mapping (cartesian to polar coordinates or kalman states to measurement states) 
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    // Radar measurement covariance donot change with time
    ekf_.R_ = R_radar_;
    // 
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // TODO: Laser updates
    // Mapping kalman states to measurement state donot change with time
    ekf_.H_ = H_laser_;
    // Laser measurement covariance donot change with time 
    ekf_.R_ = R_laser_;
    //
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
