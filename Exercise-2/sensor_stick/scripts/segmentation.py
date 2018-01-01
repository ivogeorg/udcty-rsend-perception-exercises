#!/usr/bin/env python

# Import modules
from pcl_helper import *

# TODO: Define functions as required
# TODO: Could define routines to cleanup pcl_callback
# TODO: and centralized parameters 

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)


    # Voxel grid downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # Passthrough filter
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # TODO: The table edge remains. Strategies to remove it:
    # TODO: 1. Add a 'y' passthrough filter + RANSAC removal
    # TODO: 2. Fine tune the 'z' filter limits and the RANSAC threshold
    # TODO: Note that the edge appears in both point clouds

    # Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False) # table
    extracted_outliers = cloud_filtered.extract(inliers, negative=True) # tabletop objects

    #  Create Cluster-Mask Point Cloud to visualize each cluster separately
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()

    ec = white_cloud.make_EuclideanClusterExtraction()
    # Tune these parameters
    ec.set_ClusterTolerance(0.05)  # 0.001
    ec.set_MinClusterSize(10)      # 10
    ec.set_MaxClusterSize(3000)    # 250

    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract() # a list of lists, one for each object

    # Create final point cloud where points of different lists have different colors
    cluster_colors = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, index in enumerate(indices):
            color_cluster_point_list.append([white_cloud[index][0], \
                                             white_cloud[index][1], \
                                             white_cloud[index][2], \
                                             rgb_to_float(cluster_colors[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(extracted_inliers) # table
    ros_cloud_objects = pcl_to_ros(extracted_outliers) # objects
    ros_cluster_cloud = pcl_to_ros(cluster_cloud) # colored objects

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber('/sensor_stick/point_cloud', pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher('/pcl_table', PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher('/pcl_cluster', PointCloud2, queue_size=1)

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

