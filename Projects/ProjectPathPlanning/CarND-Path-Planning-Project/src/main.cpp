#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"

// Student Code Start
#include <math.h>
#include <chrono>
#include <thread>
#include "spline.h"

using namespace std;
// Student Code End

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }


  
//     h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
//                &map_waypoints_dx,&map_waypoints_dy]
//               (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
//                uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    // Student Code Start
    // Defining lane variable, here we have 0 for leftmost lane, 1 for middle lane, 2 for rightmost lane
    int lane = 1;
    // Defining variables for Target velocity, Maximum Speed and Maximum Acceleration in m/s, m/s and m/S2 respectively
    double v_ref = 0;
    h.onMessage([&v_ref, &map_waypoints_x, &map_waypoints_y, &map_waypoints_s, &map_waypoints_dx,
    			&map_waypoints_dy, &lane]
                (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, 
                 uWS::OpCode opCode) {
    
    const double Max_Speed = 49.5; 
    const double Max_Acc = 0.224;
    // Student Code End
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          
          // Fetch number of points in previous path
          int prev_path_size = previous_path_x.size();

          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // set current s to last path s if we travelled
          if(prev_path_size > 0)
          {
            car_s = end_path_s;
          }

          // Sensor fusion Data, a list of all other cars on the same side of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];
		  
          // Student Code Start
          /**
           * TODO: define a path made up of (x,y) points that the car will visit
           *   sequentially every .02 seconds
           */
          
          vector<bool> obsinfo = GetObsInfo(sensor_fusion, prev_path_size, car_s, lane);
          bool obs_front = obsinfo[0];
          bool obs_left = obsinfo[1];
          bool obs_right = obsinfo[2];
          double delta_v = 0;
          
		  // Behavior Planning
          // Take actions based on the location of other vehicles
          if(obs_front)
          {
            // Obstacle in front
            // Check for left lane shift
            if(!obs_left && lane > 0) 
            {
              lane--; // Change lane left.
            }
            // Check for right lane shift
            else if(!obs_right && lane != 2) 
            {
              lane++; // Change lane right.
            }
            else
            {
             delta_v -= Max_Acc; // No lane change possible, decrease velocity to avoid collision
            }
          }
          else 
          {
            // Nothing in front of ego vehicle
            if(lane != 1) // Check if we are in the middle lane
            {
              if((lane == 0 && !obs_right) || (lane == 2 && !obs_left)) 
              {
                lane = 1; // Back to center.
              }
            }
            // If we are going below speed limit
            if(v_ref < Max_Speed) 
            {
              delta_v += Max_Acc; // floor it
            }
          }

          // Vectors to store waypoints, to be used to generate a smooth spline path
          vector<double> pts_x;
          vector<double> pts_y;

          // Variables storing position, orientation of the ego vehicle from last cycle
          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);

          // Loop to store waypoints from previous path for spline curve fitting
          if(prev_path_size < 2)
          // We do not have sufficient points in previous path executed, 
          // so we'll use current ego state and create an extra point
          // looking backward which is tanget to current ego yaw
          {
            // Create another point for second last cycle based on current yaw
            // and at unit distance from current position
            double prev_car_x = car_x - cos(ref_yaw); 
            double prev_car_y = car_y - sin(ref_yaw);

            // Store waypoints for spline curve fitting and avoiding jerk at the beginning
            pts_x.push_back(prev_car_x);
            pts_x.push_back(car_x);
            pts_y.push_back(prev_car_y);
            pts_y.push_back(car_y);
          }
          else
	      // The first case is, if we have sufficient points from previous cycle
          {
            ref_x = previous_path_x[prev_path_size - 1]; // update ego state x from last cycle
            ref_y = previous_path_y[prev_path_size - 1]; // update ego state y from last cycle
   
            double prev_ref_x = previous_path_x[prev_path_size - 2]; // update ego state x from second last cycle
            double prev_ref_y = previous_path_y[prev_path_size - 2]; // update ego state y from second last cycle

            // Store the last two waypoints from previous cycle for spline curve fitting and avoiding jerk at the beginning
            pts_x.push_back(prev_ref_x);
            pts_x.push_back(ref_x);
            pts_y.push_back(prev_ref_y);
            pts_y.push_back(ref_y);

         	// We also update ego vehicle yaw state
            ref_yaw = atan2(ref_y - prev_ref_y, ref_x - prev_ref_x);
          }


          // Create some more waypoints for spline path creation (30, 60, 90 mt in front from the cars current frenet coordinate position)
          vector<double> wp0 = getXY(car_s + 30, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> wp1 = getXY(car_s + 60, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> wp2 = getXY(car_s + 90, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);

          // Store the newly created waypoints
          pts_x.push_back(wp0[0]);
          pts_x.push_back(wp1[0]);
          pts_x.push_back(wp2[0]);

          pts_y.push_back(wp0[1]);
          pts_y.push_back(wp1[1]);
          pts_y.push_back(wp2[1]);

          // Transform the waypoints coordinates stored in pts_x and pts_y from inertial to car's frame of reference 
          // (The initial position of the car is that it is moving in x direction) 
          for (int i = 0; i < pts_x.size(); i++)
          {
            // Shift car reference angle to zero degrees
            double delta_x = pts_x[i] - ref_x;
            double delta_y = pts_y[i] - ref_y;

            // Update points
            pts_x[i] = delta_x*cos(0 - ref_yaw) - delta_y*sin(0 - ref_yaw);
            pts_y[i] = delta_x*sin(0 - ref_yaw) + delta_y*cos(0 - ref_yaw);
          }

          // Create a spline
          tk::spline sp;

          // Set (x,y) points for the spline
          sp.set_points(pts_x, pts_y);
          
          // Define actual waypoints we will use for the plannar
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          // Start with all of the previous path points from previous cycle
          for (int i = 0; i < prev_path_size; i++)
          {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          // Calculate how to break up spline points so that we travel at our desired ego velocity
          double target_x = 30;
          double target_y = sp(target_x);
          double target_dist = sqrt(target_x*target_x + target_y*target_y);

          double x_add_on = 0;
          
		  // Fill up rest of our plannar after filling it with previous points, here we are outputing always 50 points
          for(int i = 1; i <= 50 - prev_path_size; i++)
          {
            v_ref += delta_v;
            if ( v_ref > Max_Speed ) {
              v_ref = Max_Speed;
            } else if ( v_ref < Max_Acc ) {
              v_ref = Max_Acc;
            } 
            
            // Calculate the number of points to regulate speed
            double N = (target_x/(0.02*v_ref/2.24));

            // Waypoint coordinates in car's frame
            double x_car_frame = x_add_on + target_x/N;
            double y_car_frame = sp(x_car_frame);

            x_add_on = x_car_frame;

            // Converting waypoint coordinates from car's frame to inertial frame
            // Rotating back to normal after rotating it earilier
            double x_current = ref_x + x_car_frame * cos(ref_yaw) - y_car_frame*sin(ref_yaw);
            double y_current = ref_y + x_car_frame * sin(ref_yaw) + y_car_frame*cos(ref_yaw);

            // Push point in to trajectory vectors
            next_x_vals.push_back(x_current);
            next_y_vals.push_back(y_current);
          }
          
          json msgJson;
          
          // Student Code End


          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

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

