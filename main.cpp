
#include <ros/ros.h>
#include "cudaCode.h"
using namespace cr;

int main(int argc, char** argv)
{
	cudaCode obj;
    ros::init(argc, argv, "cudatest");
    ros::NodeHandle nh;
    ros::Rate r = 1; // Hz
	while (ros::ok())
 	{
		obj.cudaRowReduce();
	    ROS_INFO_STREAM("I am here");
	    ros::spinOnce();
	    r.sleep();
  	}

  	ROS_INFO_STREAM("Sender wird beendet"); // Sender is terminated
}

//int main(int argc, char** argv)
//{
//    ros::init(argc, argv, "cudatest");
//    ros::NodeHandle nh;
//    ROS_INFO_STREAM("I am here");
//    ros::spin();
//    return 0;
//}
