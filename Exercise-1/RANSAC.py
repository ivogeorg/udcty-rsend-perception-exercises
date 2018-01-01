# Import PCL module
import pcl

# Load Point Cloud file
cloud = pcl.load_XYZRGB('tabletop.pcd')


# FILTER 1: Voxel grid filter
# Create a VoxelGrid filter object
vox = cloud.make_voxel_grid_filter()

# Choose voxel size (aka leaf) (units are meters)
LEAF_SIZE = 0.01

# Set the leaf size for each dimension
# The resulting unit volume will be filtered to a single point
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

# Filter and save the resultant downsampled image
cloud_filtered = vox.filter()
filename = 'voxel_downsampled.pcd'
pcl.save(cloud_filtered, filename)


# FILTER 2: Pass-through filter
# Create a PassThrough object
passthrough = cloud_filtered.make_passthrough_filter()

# Assign axes and range
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.6
axis_max = 1.1
passthrough.set_filter_limits(axis_min, axis_max)

# Perform filtering to obtain resultant PC
cloud_filtered = passthrough.filter()
filename = 'pass_through_filtered.pcd'
pcl.save(cloud_filtered, filename)


# FILTER 3: Object removal via RANSAC plane segmentation
# Create a segmentation object
seg = cloud_filtered.make_segmenter()

# Set model and method to fit
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

# Max distance for a point to be considered to fit the model
max_distance = 0.01
seg.set_distance_threshold(max_distance)

# Perform segmentation to obtain inlier indices and model coeffs
inliers, coefficients = seg.segment()

# Extract inliers (table) and save pcd
extracted_inliers = cloud_filtered.extract(inliers, negative=False)
filename = 'extracted_inliers.pcd'
pcl.save(extracted_inliers, filename)

# Extract outliers (tabletop objects) and save pcd
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
filename = 'extracted_outliers.pcd'
pcl.save(extracted_outliers, filename)

