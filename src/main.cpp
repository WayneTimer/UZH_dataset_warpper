#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
backward::SignalHandling sh;
}; // namespace backward

#include <cstdio>
#include <string>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <ros/package.h>
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <boost/thread.hpp>

#include "utils.h"
#include "UZH_warpper.h"

#define DOUBLE_EPS 1e-6
#define PYRDOWN_LEVEL 3

// UZH dataset
#define WIDTH 640
#define HEIGHT 480
// 10% error in ground truth would be considered as outliers
#define OUTLIER_RATIO 0.10

// set a sensing range
#define MAX_DEPTH 12.0

char save_folder[] = "/home/timer/icra2018data/uzh_mapping_k=(3,1,0)";
char save_file[] = "/home/timer/icra2018data/uzh_mapping_k=(3,1,0)/record.txt";
FILE *record_file;


class CALI_PARA
{
public:
    double fx[PYRDOWN_LEVEL],fy[PYRDOWN_LEVEL];
    double cx[PYRDOWN_LEVEL],cy[PYRDOWN_LEVEL];
    int width[PYRDOWN_LEVEL],height[PYRDOWN_LEVEL];
};

CALI_PARA cali;
std::vector<Depth_data> depth_vec;
ros::Publisher pub_cur_pose, pub_image;
ros::Publisher pub_my_depth, pub_gt_depth, pub_error_map;
ros::Publisher pub_gt_depth_float;

std::string path_base;
UZH_warpperPtr uzh;
ros::Time time_stamp;

int idx;

double cal_density(Eigen::MatrixXd& depth)
{
    double ans = 0.0;
    int height = cali.height[0];
    int width = cali.width[0];

    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (depth(v,u) < MAX_DEPTH)
                ans = ans + 1.0;
    ans = ans / (height*width);
    return ans;
}

double cal_error(Eigen::MatrixXd& my_depth, Eigen::MatrixXd& gt_depth, Eigen::MatrixXd& error_map)
{
    double ans = 0.0;
    double cnt = 0.0;
    int height = cali.height[0];
    int width = cali.width[0];

    error_map = Eigen::MatrixXd::Zero(height,width);

    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (my_depth(v,u) < MAX_DEPTH)
            {
                error_map(v,u) = fabs(my_depth(v,u) - gt_depth(v,u)) / fabs(gt_depth(v,u));
                ans = ans + error_map(v,u);
                cnt = cnt + 1.0;
            }
            else
                error_map(v,u) = -1.0;
    ans = ans / cnt;
    return ans;
}

double cal_outlier_ratio(Eigen::MatrixXd& error_map)
{
    double ans = 0.0;
    double cnt = 0.0;
    int height = cali.height[0];
    int width = cali.width[0];

    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (error_map(v,u) > 0.0)
            {
                cnt = cnt + 1.0;
                if (error_map(v,u) > OUTLIER_RATIO)
                    ans = ans + 1.0;
            }
    ans = ans / cnt;
    return ans;
}

void show_depth(const char* window_name, Eigen::MatrixXd& depth, ros::Publisher &pub, const std_msgs::Header& header, double max_dep,
                char *img_save_path
               )
{
    int height,width;
    height = depth.rows();
    width = depth.cols();

    cv::Mat show_img = cv::Mat::zeros(height,width,CV_8UC1);
    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (depth(v,u) < max_dep)
            {
                double t = depth(v,u)*255.0 / max_dep;
                if (t<0) t = 0.0;
                if (t>255) t = 255.0;
                show_img.at<uchar>(v,u) = (uchar)t;
            }
    cv::Mat depth_img;
    cv::applyColorMap(show_img, depth_img, cv::COLORMAP_JET);

    {
        cv_bridge::CvImage out_msg;
        out_msg.header = header;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = depth_img.clone();
        pub.publish(out_msg.toImageMsg());

        if (img_save_path)
            cv::imwrite(img_save_path, depth_img);
    }
    //cv::imshow(window_name, depth_img);
}

void pub_depth(Eigen::MatrixXd& depth, ros::Publisher &pub, const std_msgs::Header& header)
{
    int height,width;
    height = depth.rows();
    width = depth.cols();

    cv::Mat result = cv::Mat::zeros(height,width,CV_32FC1);
    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            result.at<float>(v,u) = depth(v,u);

    {
        cv_bridge::CvImage out_msg;
        out_msg.header = header;
        out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        out_msg.image = result.clone();
        pub.publish(out_msg.toImageMsg());
    }
}

void show_error_map(const char* window_name, Eigen::MatrixXd &error_map, ros::Publisher &pub,
                    char *img_save_path
                   )
{
    int height,width;
    height = error_map.rows();
    width = error_map.cols();

    double max_err = -1;
    double avg_err = 0.0;
    double min_err = -1;
    int avg_cnt = 0;
    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (error_map(v,u) > 0.0)
            {
                avg_err += error_map(v,u);
                avg_cnt++;
                if (error_map(v,u) > max_err)
                    max_err = error_map(v,u);
                if (min_err<0 || error_map(v,u) < min_err)
                    min_err = error_map(v,u);
            }
    ROS_WARN("%s: %d x %d, max_error = %lf%%, min_error = %lf%%, avg_error = %lf%%",window_name, height, width, max_err*100.0, min_err*100.0, avg_err*100.0/avg_cnt);

    max_err = 0.30;   // truncated the max_err at 30% to visual

    cv::Mat show_img = cv::Mat::zeros(height,width,CV_8UC1);
    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
            if (error_map(v,u) > 0.0)
            {
                double t = error_map(v,u)*255.0 / max_err;
                if (t<0) t = 0.0;
                if (t>255) t = 255.0;
                show_img.at<uchar>(v,u) = (uchar)t;
            }
    cv::Mat depth_img;
    cv::applyColorMap(show_img, depth_img, cv::COLORMAP_JET);   //  red: big error,  blur: small error

    {
        cv_bridge::CvImage out_msg;
        out_msg.header.stamp = time_stamp;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = depth_img.clone();
        pub.publish(out_msg.toImageMsg());

        if (img_save_path)
            cv::imwrite(img_save_path, depth_img);
    }
}

void depth_callback(const sensor_msgs::ImageConstPtr msg)
{
    double t1 = msg->header.stamp.toSec();
    int i,l;
    l = depth_vec.size();

    for (i=0;i<l;i++)
    {
        double t;
        t = fabs(depth_vec[i].header.stamp.toSec() - t1);
        if (t < DOUBLE_EPS)
            break;
    }
    if (i>=l)
    {
        ROS_ERROR(" i >= l, can not find correct gt_depth in depth_vec.");
        ros::shutdown();
    }

    cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
    Eigen::MatrixXd my_depth;
    cv::cv2eigen(depth_ptr->image,my_depth);

    double density = cal_density(my_depth);
    Eigen::MatrixXd error_map;
    double error = cal_error(my_depth, depth_vec[i].depth, error_map);
    double outlier_ratio = cal_outlier_ratio(error_map);

    ROS_WARN("density = %lf%%, error = %lf%%, outlier_ratio (10%) = %lf%%", density*100.0, error*100.0, outlier_ratio*100.0);

    // find max_dep, avg_dep, min_dep
    double gt_max_dep, my_max_dep;
    double gt_avg_dep, my_avg_dep;
    double gt_min_dep, my_min_dep;
    int gt_avg_cnt, my_avg_cnt;
    gt_max_dep = my_max_dep = -1;
    gt_avg_dep = my_avg_dep = 0.0;
    gt_min_dep = my_min_dep = -1;
    gt_avg_cnt = my_avg_cnt = 0;

    int height,width;
    height = my_depth.rows();
    width = my_depth.cols();

    for (int v=0;v<height;v++)
        for (int u=0;u<width;u++)
        {
            if (my_depth(v,u) < MAX_DEPTH)
            {
                // my_depth
                my_avg_dep += my_depth(v,u);
                my_avg_cnt++;
                if (my_depth(v,u) > my_max_dep)
                    my_max_dep = my_depth(v,u);
                if (my_min_dep<0 || my_depth(v,u) < my_min_dep)
                    my_min_dep = my_depth(v,u);
            }
            if (depth_vec[i].depth(v,u) < MAX_DEPTH)
            {   
                // gt_depth
                gt_avg_dep += depth_vec[i].depth(v,u);
                gt_avg_cnt++;
                if (depth_vec[i].depth(v,u) > gt_max_dep)
                    gt_max_dep = depth_vec[i].depth(v,u);
                if (gt_min_dep<0 || depth_vec[i].depth(v,u) < gt_min_dep)
                    gt_min_dep = depth_vec[i].depth(v,u);
            }
        }
    ROS_WARN("my_depth: %d x %d, max_dep = %lf, min_dep = %lf, avg_dep = %lf", height, width, my_max_dep, my_min_dep, my_avg_dep/my_avg_cnt);
    ROS_WARN("gt_depth: %d x %d, max_dep = %lf, min_dep = %lf, avg_dep = %lf", height, width, gt_max_dep, gt_min_dep, gt_avg_dep/gt_avg_cnt);

    pub_depth(depth_vec[i].depth, pub_gt_depth_float, msg->header);

    // save data
    fprintf(record_file, "%d %lf %lf %lf %lf %lf\n", idx, density*100.0, error*100.0, outlier_ratio*100.0, my_avg_dep/my_avg_cnt, gt_avg_dep/gt_avg_cnt);
    fflush(record_file);
    char img_save_path[100];

    sprintf(img_save_path,"%s/%d_my_depth.png", save_folder, idx);
    show_depth("my_depth", my_depth, pub_my_depth,msg->header, MAX_DEPTH, img_save_path);

    sprintf(img_save_path,"%s/%d_gt_depth.png", save_folder, idx);
    show_depth("gt_depth", depth_vec[i].depth, pub_gt_depth, msg->header, MAX_DEPTH,img_save_path);

    sprintf(img_save_path,"%s/%d_error_map.png", save_folder, idx);
    show_error_map("error_map",error_map,pub_error_map,img_save_path);
    //cv::waitKey(0);
}

void main_thread()
{
    depth_vec.clear();

    for (int i=1;i<=200;i++)
    {
        time_stamp = time_stamp + ros::Duration(0.2);
        uzh->set_next_header_stamp(time_stamp);
        pub_cur_pose.publish(uzh->next_pose(idx));
        
        Depth_data depth_data = uzh->next_depth();
        depth_vec.push_back(depth_data);
        
        pub_image.publish(uzh->next_img());
        printf("i = %d, idx = %d\n", i, idx);

        //getchar();
        sleep(1.0);
    }

    fclose(record_file);
    puts("Done with fclose()");    
}

// all depth should be in Euclid distance
int main(int argc, char **argv)
{
    ros::init(argc,argv,"UZH_dataset_warpper");
    ros::NodeHandle nh("~");

    path_base = readParam<std::string>(nh, "dataset_path_base");
    cali.width[0] = WIDTH;
    cali.height[0] = HEIGHT;

    uzh = UZH_warpperPtr(new UZH_warpper(path_base, 640, 480));
    time_stamp = ros::Time::now();

    pub_cur_pose = nh.advertise<geometry_msgs::PoseStamped>("cur_pose",100);
    pub_image = nh.advertise<sensor_msgs::Image>("image",100);
    pub_gt_depth_float = nh.advertise<sensor_msgs::Image>("gt_depth_float",100);
    pub_gt_depth = nh.advertise<sensor_msgs::Image>("gt_depth_visual",100);
    pub_my_depth = nh.advertise<sensor_msgs::Image>("my_depth_visual",100);
    pub_error_map = nh.advertise<sensor_msgs::Image>("error_map_visual",100);

    ros::Subscriber sub_depth = nh.subscribe("/motion_stereo_left/depth/image_raw",100,depth_callback);

    record_file = fopen(save_file,"w");

    boost::thread th1(main_thread);
 
    while (ros::ok())
    {
        ros::spinOnce();
    }

    //main_thread();

    return 0;
}
