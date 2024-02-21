
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
import open3d as o3d
import os
import pandas as pd


class LidarDataAugmentationNode(Node):

    def __init__(self):
        super().__init__('lidar_data_augmentation_node')
        self.ros_subscriber_topic = '/carla/ego_vehicle/semantic_lidar'
        self.ros_publisher_topic = '/bisalgt/lidar_augmented_data'
        self.subscription = self.create_subscription(
            PointCloud2,
            self.ros_subscriber_topic,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.publisher_ = self.create_publisher(PointCloud2, self.ros_publisher_topic, 10)
        # self.timer = self.create_timer(0.5, self.timer_callback)

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: data {type(msg.data)}')
        fields = [field.name for field in msg.fields]
        fields_str = ','.join(fields)
        print(fields_str)
        pcd_data = point_cloud2.read_points(msg, skip_nans=True, field_names=fields)
        self.points = np.array(list(pcd_data))[:, :3]
        self.msg = msg
        self.timer_callback()
    
    def timer_callback(self):
        print("timer_callback called")
        print(os.getcwd())

        # region Thesis Work
        self.transform_mesh()
        self.raycasting()
        self.shadow_casting()
        self.create_final_cloud()
        # endregion of Thesis Work
        print("print points shape : ",self.points.shape)
        self.msg.data = np.array(self.points, dtype=np.float32).tobytes()
        self.msg.width = len(self.points)
        self.msg.height = 1
        self.msg.point_step = 12
        self.msg.row_step = self.msg.point_step * self.msg.width
        self.msg.is_dense = False
        self.publisher_.publish(self.msg)
        self.get_logger().info('Publishing: "%s"' % type(self.msg.data))
        print("timer_callback done")
    
    def transform_mesh(self):
        min_xy = (5,5)
        max_xy = (6,6)
        self.centroid_of_reference_roi = np.asarray([ 3.86239466, -7.16583308, -2.29860912]) # precalculated when extracting the mesh from its source
        target_region = self.points[(self.points[:, 0] >= min_xy[0]) & (self.points[:, 0] <= max_xy[0]) & (self.points[:, 1] >= min_xy[1]) & (self.points[:, 1] <= max_xy[1])]
        self.centroid_of_target_roi = np.mean(target_region, axis=0)
        print(self.centroid_of_target_roi)
        print(self.centroid_of_reference_roi)
        translation = self.centroid_of_target_roi - self.centroid_of_reference_roi
        translation_matrix = np.identity(4)
        # print(translation_matrix)
        # print(translation)
        # print(translation_matrix[:3, 3])
        translation_matrix[:3, 3] = translation
        self.person_mesh = o3d.io.read_triangle_mesh("src/lidar_augmentation_node/lidar_augmentation_node/person_mesh.ply")
        self.person_mesh.transform(translation_matrix)

        # rotation part

        start_vector = np.array(self.centroid_of_reference_roi)
        end_vector = np.array(self.centroid_of_target_roi)
        # Calculate the cosine of the angle
        cos_angle = np.dot(start_vector, end_vector) / (np.linalg.norm(start_vector) * np.linalg.norm(end_vector))
        # Calculate the sine of the angle
        sin_angle = np.linalg.norm(np.cross(start_vector, end_vector)) / (np.linalg.norm(start_vector) * np.linalg.norm(end_vector))
        # Calculate the angle in radians
        angle_rad = np.arctan2(sin_angle, cos_angle)
        # If the z-component of the cross product is negative, the angle should be negative
        if np.cross(start_vector, end_vector)[2] < 0:
            print("Negative angle")
            angle_rad = -angle_rad
        else:
            print("Positive angle")
        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)
        # Create the rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        print("Rotation Matrix: ", rotation_matrix)
        self.person_mesh.rotate(rotation_matrix, center=np.asarray(self.person_mesh.get_center()))

    
    def raycasting(self):
        self.transform_mesh() # transform the person mesh to the target region
        # rays creation
        ray_origin = np.zeros((len(self.points), 3))
        rays_array = np.concatenate((ray_origin, self.points[:, :3]), axis=1)
        self.rays = o3d.core.Tensor(rays_array,dtype=o3d.core.Dtype.Float32)
        # raycasting
        mesh_new = o3d.t.geometry.TriangleMesh.from_legacy(self.person_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_new)
        self.after_raycast_result = scene.cast_rays(self.rays)

        # calculation of the intersection point
        index_of_intersected_triangles = np.where(self.after_raycast_result["primitive_ids"].numpy() != scene.INVALID_ID)[0]
        intersected_rays = rays_array[index_of_intersected_triangles]
        t_hit = self.after_raycast_result["t_hit"].numpy()[index_of_intersected_triangles]
        df = pd.DataFrame(intersected_rays, columns=["x0", "y0", "z0", "x1", "y1", "z1" ])
        distance = ((df["x0"] - df["x1"])**2 + (df["y0"] - df["y1"])**2 + (df["z0"] - df["z1"])**2)**(1/2)
        df["distance"] = distance
        df["t_hit"] = t_hit
        df["x"] = df["x0"] + df["t_hit"] * (df["x1"] - df["x0"])
        df["y"] = df["y0"] + df["t_hit"] * (df["y1"] - df["y0"])
        df["z"] = df["z0"] + df["t_hit"] * (df["z1"] - df["z0"])
        self.raycasted_prototype = df[["x", "y", "z"]].values
        print("raycasting done")


    def shadow_casting(self):
        t_hits_all = self.after_raycast_result["t_hit"].numpy()
        not_hits_indices = np.where(t_hits_all == np.inf)[0]
        self.region_without_shadow = self.points[not_hits_indices]
        print("shadow casting done")

    def create_final_cloud(self):
        self.points = np.concatenate((self.raycasted_prototype, self.region_without_shadow), axis=0)
        


def main(args=None):
    rclpy.init(args=args)

    lidar_data_augmentation_node = LidarDataAugmentationNode()

    rclpy.spin(lidar_data_augmentation_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    lidar_data_augmentation_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
