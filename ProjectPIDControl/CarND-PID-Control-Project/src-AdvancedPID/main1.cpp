// Student Code Start
#define _USE_MATH_DEFINES
// Student Code End
#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>
#include <string>

// Student Code Start
#include <stdlib.h>
using namespace std;
using std::string;
// For convenience
using nlohmann::json;
using json = nlohmann::json;
// Student Code Ends

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != string::npos) {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main() {
  uWS::Hub h;

  PID pidSteer;
  PID pidSpeed;
  
  // Student Code Start

  /**
   * TODO: Initialize the pid variable.
   */
  // Using Init function of PID class to initialise Kp, Ki, Kd.
  // pid.Init(0.1, 0.001, 1);

  // h.onMessage([&pid](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, 
  //                    uWS::OpCode opCode) {
  
  h.onMessage([&pidSteer, &pidSpeed](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, 
                     uWS::OpCode opCode) {

    // Choose whether Constant speed or PID controlled speed
    bool speed_const = false;
    // Select max speed for PID controller or Constant speed value
    double dSpeed_max = 95.;
    double dSpeed_const = 0.75;

    // Initialize the pid variable.

    if (!pidSteer.is_initialized) {
      // Initialised Kp Ki Kd parameters, like p in lectures
      pidSteer.vc_dKparam = {3.2, 170, 0.0002} ;
      // pidSteer.vc_dKparam = {2.0, 80.0, 0.0002} ;
      // The increment interval, like dp in lectures
      pidSteer.vc_dKtune = { 1.0, 20.0, 0.0005} ;
      // Set the flag to do Twiddle or Not
      pidSteer.do_twiddle = false;
      // If we donot do twiddle above, then just use the Initialised Kp, Ki and Kd parameters for PID control
      pidSteer.Init(pidSteer.vc_dKparam[0], pidSteer.vc_dKparam[1], pidSteer.vc_dKparam[2]);
      //cout << "\n start steer parameter \n" << pidSteer.d_Kp << "\t" << pidSteer.d_Kd << "\t" << pidSteer.d_Ki << "\t" << pidSteer.iPID_opt << "\n";
    }

    if (!pidSpeed.is_initialized) {
      // Initialised Kp Ki Kd parameters, like p in lectures
      pidSpeed.vc_dKparam = { 18.0, 20.0, 0.002} ;
      // The increment interval, like dp in lectures
      pidSpeed.vc_dKtune = { 1., 1.0, 0.0005};
      // Set the flag to do Twiddle or Not
      pidSpeed.do_twiddle = false;
      // If we donot do twiddle above, then just use the Initialised Kp, Ki and Kd parameters for PID control
      pidSpeed.Init(pidSpeed.vc_dKparam[0], pidSpeed.vc_dKparam[1], pidSpeed.vc_dKparam[2]);
      //cout << "\n start speed parameter \n" << pidSpeed.d_Kp << "\t" << pidSpeed.d_Kd << "\t" << pidSpeed.d_Ki << "\t" << pidSpeed.iPID_opt << "\n";
    }

    // Student Code End
    
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(string(data).substr(0, length));

      if (s != "") {
        auto j = json::parse(s);
        // Studen Code Start
        // Do steer twiddle when number of steps is reached (e.g. one lap)
        if(pidSteer.do_twiddle && pidSteer.i_n_steps == pidSteer.iPID_opt){
          std::string event = j[0].get<std::string>();
          if (event == "telemetry") {
            // j[1] is the data JSON object
            double cte = std::stod(j[1]["cte"].get<string>());
                      /**
             * TODO: Calculate steering value here, remember the steering value is
             *   [-1, 1].
             * NOTE: Feel free to play around with the throttle and speed.
             *   Maybe use another PID controller to control the speed!
             */
            
            // Update errors and Calculate the steer_value
            cte = pow(cte,2)*cte/fabs(cte);
            pidSteer.UpdateError(cte);
            
            // Do twiddle loop here, to find better Kp, Ki, Kd
            pidSteer.Twiddle_param();

            pidSteer.i_n_steps = 1;
            pidSteer.norm_steer_value = 0;
            cte = 0.;

            // Reset vehicle to start point and condition
            json msgJson;
            std::string reset_msg = "42[\"reset\",{}]";
            ws.send(reset_msg.data(), reset_msg.length(), uWS::OpCode::TEXT);
          } // Do speed twiddle when number of steps is reached (e.g. one lap)

        } else if (pidSpeed.do_twiddle && pidSpeed.i_n_steps == pidSpeed.iPID_opt){

          std::string event = j[0].get<std::string>();
          if (event == "telemetry") {

            // j[1] is the data JSON object
            double speed = std::stod(j[1]["speed"].get<std::string>());
            double angle = std::stod(j[1]["steering_angle"].get<std::string>());
            // double dSpeed_max = 100.;

            pidSpeed.d_Speed_sum +=speed;

            double ct_error;
            // calculate a cte with speed difference and a function pow(x,4)=> forcing lower speed at higher angles
            ct_error = -1*pow((speed - dSpeed_max)/10.,2) + pow(fabs(angle)/4.7,4)/15.;
            pidSpeed.UpdateError(ct_error);
            pidSpeed.d_mean_squared_error = 100. - pidSpeed.d_Speed_sum  / pidSpeed.i_n_steps;
            //cout<<"d_mean_squared_error d_Speed_sum: \t" <<  pidSpeed.d_mean_squared_error <<"\n";
            // twiddle loop here
            pidSpeed.Twiddle_param();

            pidSpeed.i_n_steps = 1;
            pidSpeed.norm_steer_value = 0;
            speed = 0.;
            angle = 0.;
            // reset vehicle to start point and condition
            json msgJson;
            std::string reset_msg = "42[\"reset\",{}]";
            ws.send(reset_msg.data(), reset_msg.length(), uWS::OpCode::TEXT);
          }        
          // no twiddle, just cruise around
          } else {

              string event = j[0].get<string>();
              if (event == "telemetry") {

                // j[1] is the data JSON object    
                double cte = std::stod(j[1]["cte"].get<string>());
                double speed = std::stod(j[1]["speed"].get<string>());
                double angle = std::stod(j[1]["steering_angle"].get<string>());
                double steer_value, speed_value;
                // double dSpeed_max = 100.;

                pidSpeed.d_Speed_sum +=speed;
                
                // Calculate the cte
                cte = pow(cte,2)*cte/fabs(cte);
                
                // Update errors and calculate the steer_value
                pidSteer.UpdateError(cte);
                steer_value =pidSteer.TotalError();

                // Normalize steer value to [-1,1]
                pidSteer.norm_steer_value = steer_value/(pidSteer.d_steer_max) ;
                
                if(fabs(pidSteer.norm_steer_value)>1.0){
                  pidSteer.norm_steer_value /=fabs(pidSteer.norm_steer_value);
                }

                // Speed value
                double ct_error;
                ct_error = -1*pow((speed - dSpeed_max)/10.,2) + pow(fabs(angle)/4.7,4)/15.;

                // Update speed errors
                pidSpeed.UpdateError(ct_error);
                speed_value = pidSpeed.TotalError();

                pidSpeed.norm_steer_value = speed_value;

                // Normalize speed value to [0,1]
                if(pidSpeed.norm_steer_value < 0.0 ){
                  pidSpeed.norm_steer_value = 0.0;
                } else if (pidSpeed.norm_steer_value > dSpeed_max) {
                  pidSpeed.norm_steer_value = dSpeed_max/100.;
                }else {
                  pidSpeed.norm_steer_value = pidSpeed.norm_steer_value / 100.;
                }

                // DEBUG
                //std::cout << "CTE: " << cte << " Steering Value: " << pidSteer.norm_steer_value << "  Roh steering " << steer_value << std::endl;

                json msgJson;
                msgJson["steering_angle"] = pidSteer.norm_steer_value;
                if(speed_const){
                  msgJson["throttle"] = dSpeed_const;
                }else {
                  msgJson["throttle"] =  pidSpeed.norm_steer_value;
                }

                auto msg = "42[\"steer\"," + msgJson.dump() + "]";
                //std::cout << msg << std::endl;
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);         
              }
            } 
         } else { 
              // Manual driving
              string msg = "42[\"manual\",{}]";
              ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
    }// end websocket message if
  }); // end h.onMessage


  // Student Code Start
  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  // h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
  //   const std::string s = "<h1>Hello world!</h1>";
  //   if (req.getUrl().valueLength == 1)
  //   {
  //     res->end(s.data(), s.length());
  //   }
  //   else
  //   {
  //     // i guess this should be done more gracefully?
  //     res->end(nullptr, 0);
  //   }
  // });
  // Student Code End


  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, 
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}