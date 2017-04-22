#include "UZH_warpper.h"

UZH_warpper::UZH_warpper(std::string _path_base, int _width, int _height)
{
    std::string str;
    path_base = _path_base;

    str = path_base + "/info/depthmaps.txt";
    depth_info = fopen(str.c_str(), "r");
    if (!depth_info)
    {
        ROS_ERROR("depth_info file open ERROR !");
    }

    str = path_base + "/info/groundtruth.txt";
    gt_info = fopen(str.c_str(), "r");
    if (!gt_info)
    {
        ROS_ERROR("gt_info file open ERROR !");
    }

    str = path_base + "/info/images.txt";
    img_info = fopen(str.c_str(), "r");
    if (!img_info)
    {
        ROS_ERROR("img_info file open ERROR");
    }

    width = _width;
    height = _height;
}

void UZH_warpper::set_next_header_stamp(ros::Time &stamp)
{
    header.stamp = stamp;
}

geometry_msgs::PoseStamped UZH_warpper::next_pose()
{
    geometry_msgs::PoseStamped pose;

    int cnt;
    double tx,ty,tz,qx,qy,qz,qw;

    fscanf(gt_info, "%d %lf %lf %lf %lf %lf %lf %lf",
                     &cnt,
                     &tx, &ty, &tz,
                     &qx, &qy, &qz, &qw);

    pose.header = header;
    pose.pose.position.x = tx;
    pose.pose.position.y = ty;
    pose.pose.position.z = tz;
    pose.pose.orientation.x = qx;
    pose.pose.orientation.y = qy;
    pose.pose.orientation.z = qz;
    pose.pose.orientation.w = qw;

    ROS_INFO("%d read done!  pose: (%.2lf, %.2lf, %.2lf), (x,y,z,w = (%.2lf,%.2lf,%.2lf,%.2lf))",
              cnt, pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
              pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w
            );

    return pose;
}

sensor_msgs::ImagePtr UZH_warpper::next_img()
{
    char line[100];
    std::string file_name;
    int cnt;
    double ts;
    fscanf(img_info,"%d %lf %s", &cnt, &ts, line);
    file_name = path_base + "/data/" + line;

    cv_bridge::CvImage img_msg;
    img_msg.header = header;
    img_msg.encoding = sensor_msgs::image_encodings::MONO8;
    img_msg.image = cv::imread(file_name,0);
    return img_msg.toImageMsg();
}

Depth_data UZH_warpper::next_depth()
{
    char line[100];
    std::string file_name;
    int cnt;
    fscanf(depth_info,"%d %s", &cnt, line);
    file_name = path_base + "/data/" + line;

    FILE *depth_file = fopen(file_name.c_str(),"r");

    depth.depth = Eigen::MatrixXd::Zero(height,width);

    // The depths are stored in row major order.
    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
        {
            double tmp;
            fscanf(depth_file, "%lf", &tmp);
            depth.depth(v,u) = tmp;     // depth_euclid: depth along the ray
        }

    depth.header = header;

    return depth;
}
