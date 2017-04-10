#ifndef _UZH_warpper_h
#define _UZH_warpper_h

#include <string>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <ros/package.h>
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

class Depth_data
{
public:
    Eigen::MatrixXd depth;
    std_msgs::Header header;
};

class UZH_warpper
{
public:
    UZH_warpper(std::string _path_base, int _width, int _height);
    void set_next_header_stamp(ros::Time &stamp);
    geometry_msgs::PoseStamped next_pose();
    sensor_msgs::ImagePtr next_img();
    Depth_data next_depth();

    int width,height;
    Depth_data depth;

private:
    std::string path_base;
    FILE *depth_info, *gt_info, *img_info;
    std_msgs::Header header;
};

typedef boost::shared_ptr<UZH_warpper> UZH_warpperPtr;

#endif
