#include "PID.h"
// Student Code Start
using namespace std;
// Student Code End
/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

// Student Code Start
// PID::PID() {
//  p_error = 0;
// 	i_error = 0;
// 	d_error = 0;
// }
PID::PID() : is_initialized(false) {};
// Student Code End

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */

  // Student Code Start
  // Initialising errors for Kp, Ki and Kd
  d_p_error = 0.0;
  d_i_error = 0.0;
  d_d_error = 0.0;

  // Kp = Kp_;
  // Ki = Ki_;
  // Kd = Kd_;

  d_Kp = Kp_;
  d_Kd = Kd_;
  d_Ki = Ki_;

  d_total_error = 0.0;

  // Maximum steering variable, to be used lated to normalize steering angle between (-25° - 25°)
  d_steer_max = 25.0;

  // Twiddle algorithm parameter
  dTol = 0.1;
  iSwitch_ = 0;
  d_abs_error = 0.;
  d_squared_error = 0.;
  d_mean_squared_error = 0.;
  d_best_error = std::numeric_limits<double>::max();
  i_n_steps = 1;

  // When we want to do twiddle use the following flag.
  // do_twiddle = true;
  iPID_opt = 4000;
  iN_max_steps = 4000;
  dN_min_Tol = 0.001;
  iTwd_iter= 1;
  i_iter = 2;
  k = 0;

  d_Kp_best = Kp_;
  d_Kd_best = Kd_;
  d_Ki_best = Ki_;

  d_Speed_sum = 0.0;

  // Student Code End

}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  // Student Code Start
 
  if (i_n_steps == 1) {
      // If loop is for getting correct initial d_error
          d_pre_cte = cte;
  }

	// i_error += cte;
  // // We need to write p_error after calculating d_error, as we need old p_error in below formula
	// d_error = cte - p_error;
	// p_error = cte;

  // Calculate Kp, Ki and Kd error
  d_p_error = cte;
  d_d_error = cte - d_pre_cte;
  d_i_error += cte;

  // Set d_pre_cte, so that it can be used to calculate d_d_error, in the next iteration
  d_pre_cte = cte;

  // Calculate absolute, squared and mean squared error
  d_abs_error +=fabs(cte) ;
  d_squared_error +=pow(cte,2);
  d_mean_squared_error = d_squared_error / i_n_steps;

  // Summing over number of steps taken to use in calculating mean_squared_error
  i_n_steps +=1;
  
  // Student Code End

}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  // Student Code Start

  // Note here we use d_Kp, d_Kd and d_Ki, these gain parameters we get from twiddle algorithm
  d_total_error = - d_Kp * d_p_error
                 - d_Kd * d_d_error
                 - d_Ki * d_i_error;
  return d_total_error;
  // return Kp * p_error + Ki * i_error + Kd * d_error;
  // return 0.0;  // TODO: Add your total error calc here!
  
  // Student Code End
}

 /*
  * Twiddle
  */
  void PID::Twiddle_param(){
    if(accumulate(begin(vc_dKtune), end(vc_dKtune), 0.0, plus<double>())>dTol){

      if(d_mean_squared_error < d_best_error){
        d_best_error = d_mean_squared_error;
        d_Kp_best =  d_Kp;
        d_Kd_best =  d_Kd;
        d_Ki_best =  d_Ki;
        iSwitch_ =1;
      } else {
        iSwitch_ +=2;
      }

      cout<<"\n Iteration " <<  iTwd_iter << "\t" << "best error = " << d_best_error <<"\n";
      cout<<"Best parameter set: \n ";
      cout<< d_Kp_best <<"\t"<< d_Kd_best << "\t"<<d_Ki_best <<"\n";
      cout<<"Actual parameter set: \n ";
      cout<< d_Kp <<"\t"<< d_Kd << "\t"<<d_Ki <<"\n";
      cout<< d_abs_error <<"\t"<< d_squared_error << "\t"<<d_mean_squared_error <<"\n";

      switch(iSwitch_)
      {
        case 1:
          k = i_iter%3;
          vc_dKtune[k] *=1.1;
          iSwitch_ = 0;
          iTwd_iter += 1;
          i_iter +=1;
          k = i_iter%3;
          vc_dKparam[k]+=vc_dKtune[k];
          Twiddle_reset();
          break;
        case 2:
          k = i_iter%3;
          vc_dKparam[k]-=2*vc_dKtune[k];
          Twiddle_reset();
          iTwd_iter += 1;
          break;
        case 4:
          k = i_iter%3;
          vc_dKparam[k]+=vc_dKtune[k];
          vc_dKtune[k] *=0.9;
          iSwitch_ = 0;
          iTwd_iter += 1;
          i_iter +=1;
          k = i_iter%3;
          vc_dKparam[k]+=vc_dKtune[k];
          Twiddle_reset();
          break;
      }
    } else {
      if (iPID_opt<iN_max_steps && dTol>dN_min_Tol){
        iPID_opt  +=1000;
        dTol /=10;
        double dTune = dTol*100. ;
        vc_dKtune = {dTune, dTune, dTune};
        cout<< vc_dKparam[0] <<"\t" << vc_dKparam[1] <<"\t" << vc_dKparam[2] <<"\n" ;
        Twiddle_reset();
        i_iter = 2;
        iTwd_iter += 1;
      } else {

        cout<<"\n Iteration " <<  iTwd_iter << "\t" << "best error = " << d_best_error <<"\n";
        cout<<"\n Best Parameters  Kp: " << d_Kp_best <<"\t Kd: "<< d_Kd_best << "\t  Kd: "<<d_Ki_best <<"\n";
        do_twiddle = false;
        exit(0);
      }
    }
  }

  /*
  * Restart Twiddle-loop
  */
  void PID::Twiddle_reset(){
    cout<<"Twiddle_reset \n";
    d_Kp =  vc_dKparam[0];
    d_Kd =  vc_dKparam[1];
    d_Ki =  vc_dKparam[2];
    d_i_error = 0.;
    d_abs_error =0.;
    d_squared_error =0.;
    d_mean_squared_error = 0.;
    norm_steer_value = 0.;
    d_Speed_sum = 0.0;
   }

