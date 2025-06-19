#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
from tf2_ros import TransformBroadcaster

class ArucoRecognition(Node):
    def __init__(self):
        super().__init__('aruco_recognition')

        # --- 1. Declaração dos Parâmetros ---
        self.declare_parameter('marker_length', 0.05)
        self.declare_parameter('dictionary', 'DICT_5X5_250')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('marker_frame_prefix', 'aruco_')
        self.declare_parameter('stability_filter.position_threshold', 0.02)
        self.declare_parameter('stability_filter.angle_threshold', 0.15)
        self.declare_parameter('camera_matrix', [1000.0, 0.0, 320.0, 0.0, 1000.0, 240.0, 0.0, 0.0, 1.0])
        self.declare_parameter('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('subscribers.image_rgb', 'camera1/image_raw')
        self.declare_parameter('publishers.recognition_image', 'aruco/detection')
        self.declare_parameter('publishers.poses', 'aruco/poses')
        self.declare_parameter('publishers.markers', 'aruco/markers')

        # --- 2. Obtenção dos Parâmetros e Configuração ---
        self.load_parameters()
        self.setup_aruco()
        self.setup_tools_and_comms()
        self.get_logger().info("ArucoRecognition node inicializado com sucesso.")

    def load_parameters(self):
        """Carrega todos os parâmetros para atributos da classe."""
        self.marker_length = float(self.get_parameter('marker_length').value)
        self.camera_frame = self.get_parameter('camera_frame').value
        self.marker_frame_prefix = self.get_parameter('marker_frame_prefix').value
        self.pos_threshold = self.get_parameter('stability_filter.position_threshold').value
        self.ang_threshold = self.get_parameter('stability_filter.angle_threshold').value
        self.camera_matrix = np.array(self.get_parameter('camera_matrix').value, dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(self.get_parameter('dist_coeffs').value, dtype=np.float32)
        self.topic_sub_camera = self.get_parameter('subscribers.image_rgb').value
        self.topic_pub_detection = self.get_parameter('publishers.recognition_image').value
        self.topic_pub_poses = self.get_parameter('publishers.poses').value
        self.topic_pub_markers = self.get_parameter('publishers.markers').value

    def setup_aruco(self):
        """Configura o detector ArUco com base nos parâmetros."""
        dictionary_name = self.get_parameter('dictionary').value
        try:
            aruco_dict_id = cv2.aruco.__getattribute__(dictionary_name)
            self.get_logger().info(f"Usando o dicionário ArUco: {dictionary_name}")
        except AttributeError:
            self.get_logger().error(f"Dicionário ArUco '{dictionary_name}' não encontrado! Usando 'DICT_5X5_250'.")
            aruco_dict_id = cv2.aruco.DICT_5X5_250

        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_id)
        if hasattr(aruco, 'DetectorParameters_create'):
            self.aruco_parameters = aruco.DetectorParameters_create()
            self.use_aruco_detector_class = False
        else:
            self.aruco_parameters = aruco.DetectorParameters()
            self.use_aruco_detector_class = hasattr(aruco, 'ArucoDetector')
        if self.use_aruco_detector_class:
            self.aruco_recognition = aruco.ArucoDetector(self.aruco_dict, self.aruco_parameters)
        else:
            self.aruco_recognition = None

    def setup_tools_and_comms(self):
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_valid_poses = {}
        qos_depth = 10
        self.image_sub = self.create_subscription(Image, self.topic_sub_camera, self.image_callback, qos_depth)
        self.detection_pub = self.create_publisher(Image, self.topic_pub_detection, qos_depth)
        self.poses_pub = self.create_publisher(PoseArray, self.topic_pub_poses, qos_depth)
        self.markers_pub = self.create_publisher(MarkerArray, self.topic_pub_markers, qos_depth)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Erro ao converter imagem: {str(e)}')
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Compatibilidade com diferentes versões do OpenCV
        if self.aruco_recognition is not None:
            corners, ids, rejected = self.aruco_recognition.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_parameters)
        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            axis_length = self.marker_length * 0.5
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], axis_length)
            self.handle_detected_markers(ids, rvecs, tvecs, msg.header)
        self.publish_detection_image(frame, msg.header)

    def handle_detected_markers(self, ids, rvecs, tvecs, header):
        pose_array = PoseArray()
        pose_array.header = header
        pose_array.header.frame_id = self.camera_frame
        marker_array = MarkerArray()
        
        for i, marker_id_raw in enumerate(ids):
            marker_id = int(marker_id_raw[0])
            current_pos, current_rot = tvecs[i][0], rvecs[i][0]
            
            if self.should_update_pose(marker_id, current_pos, current_rot):
                self.last_valid_poses[marker_id] = {'position': current_pos, 'rotation': current_rot}
            
            if marker_id in self.last_valid_poses:
                pose_array.poses.append(self.create_pose_msg(marker_id))
                marker_array.markers.append(self.create_marker_msg(marker_id, header))
        
        if pose_array.poses:
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(marker_array)
            self.publish_transforms(header)

    def should_update_pose(self, marker_id, current_pos, current_rot):
        if marker_id not in self.last_valid_poses:
            return True
        last_pos, last_rot = self.last_valid_poses[marker_id]['position'], self.last_valid_poses[marker_id]['rotation']
        pos_diff = np.linalg.norm(current_pos - last_pos)
        rot_diff = np.linalg.norm(current_rot - last_rot)
        return (pos_diff > self.pos_threshold) or (rot_diff > self.ang_threshold)

    def create_pose_msg(self, marker_id):
        pose_data = self.last_valid_poses[marker_id]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, pose_data['position'])
        rot_mat, _ = cv2.Rodrigues(pose_data['rotation'])
        q = self.rotation_matrix_to_quaternion(rot_mat)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        return pose

    def create_marker_msg(self, marker_id, header):
        """Cria uma mensagem Marker com valores de visualização fixos."""
        marker = Marker()
        marker.header = header
        marker.ns = "aruco_markers"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = self.create_pose_msg(marker_id)
        marker.scale.x = self.marker_length
        marker.scale.y = self.marker_length
        marker.scale.z = self.marker_length * 0.1
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.0, 1.0, 0.0, 0.7)
        marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
        return marker

    def publish_transforms(self, header):
        transforms = []
        for marker_id, pose_data in self.last_valid_poses.items():
            t = TransformStamped()
            t.header = header
            t.header.frame_id = self.camera_frame
            t.child_frame_id = f"{self.marker_frame_prefix}{marker_id}"
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, pose_data['position'])
            rot_mat, _ = cv2.Rodrigues(pose_data['rotation'])
            q = self.rotation_matrix_to_quaternion(rot_mat)
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
            transforms.append(t)
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def publish_detection_image(self, frame, header):
        try:
            detection_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            detection_msg.header = header
            self.detection_pub.publish(detection_msg)
        except Exception as e:
            self.get_logger().error(f'Erro ao publicar imagem de detecção: {str(e)}')

    def rotation_matrix_to_quaternion(self, rot_mat):
        m = rot_mat[:3,:3]
        tr = np.trace(m)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    detector = ArucoRecognition()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info('Nó encerrado manualmente pelo usuário (Ctrl+C).')
    finally:
        detector.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()