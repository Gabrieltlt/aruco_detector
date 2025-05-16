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

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        
        # Parâmetros
        self.declare_parameter('marker_length', 0.05)  # em metros
        self.declare_parameter('dictionary', 'DICT_5X5_250')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('marker_frame_prefix', 'aruco_')
        
        # Parâmetros da câmera (ajuste para sua câmera)
        self.declare_parameter('camera_matrix', [1000.0, 0.0, 320.0, 
                                               0.0, 1000.0, 240.0, 
                                               0.0, 0.0, 1.0])
        self.declare_parameter('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Obter parâmetros
        self.marker_length = self.get_parameter('marker_length').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.marker_frame_prefix = self.get_parameter('marker_frame_prefix').value
        
        # Configurar ArUco
        self.setup_aruco()
        
        # Configurar matriz da câmera
        self.camera_matrix = np.array(
            self.get_parameter('camera_matrix').value, 
            dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(
            self.get_parameter('dist_coeffs').value, 
            dtype=np.float32)
        
        # Bridge para OpenCV
        self.bridge = CvBridge()
        
        # TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscriber para imagens da câmera
        self.image_sub = self.create_subscription(
            Image,
            'camera1/image_raw',
            self.image_callback,
            10)
        
        # Publishers
        self.detection_pub = self.create_publisher(Image, 'aruco/detection', 10)
        self.poses_pub = self.create_publisher(PoseArray, 'aruco/poses', 10)
        self.markers_pub = self.create_publisher(MarkerArray, 'aruco/markers', 10)
        
        self.get_logger().info("Aruco Detector inicializado - Publicando imagens continuamente")

    def setup_aruco(self):
        """Configurar dicionário ArUco"""
        dictionary_name = self.get_parameter('dictionary').value
        
        aruco_dict = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
            'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
            'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
            'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
        }.get(dictionary_name, cv2.aruco.DICT_5X5_250)
        
        self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
        self.parameters = cv2.aruco.DetectorParameters_create()

    def image_callback(self, msg):
        """Callback para processar cada frame recebido"""
        try:
            # Converter mensagem ROS para imagem OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Erro ao converter imagem: {str(e)}')
            return
            
        # Criar cópia da imagem original para desenhar os marcadores
        processed_frame = frame.copy()
        
        # Processar frame para detecção de ArUco
        markers_detected = self.process_frame(processed_frame)
        
        # Publicar imagem processada (com ou sem marcadores)
        self.publish_detection(processed_frame, msg.header)
        
        # Se marcadores foram detectados, publicar informações adicionais
        if markers_detected:
            corners, ids, rvecs, tvecs = markers_detected
            self.publish_poses(ids, rvecs, tvecs, msg.header)
            self.publish_markers(ids, tvecs, msg.header)
            self.publish_transforms(ids, rvecs, tvecs, msg.header)

    def process_frame(self, frame):
        """Processar frame para detecção de ArUco e desenhar marcadores se encontrados"""
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar marcadores
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )
        
        if ids is not None:
            # Estimar pose dos marcadores
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )
            
            # Desenhar resultados no frame
            self.draw_markers(frame, corners, ids, rvecs, tvecs)
            
            return corners, ids, rvecs, tvecs
        
        return None

    def draw_markers(self, frame, corners, ids, rvecs, tvecs):
        """Desenhar marcadores e eixos no frame"""
        for i in range(len(ids)):
            # Desenhar eixos de referência
            cv2.drawFrameAxes(
                frame, self.camera_matrix, self.dist_coeffs,
                rvecs[i], tvecs[i], self.marker_length
            )
            
            # Exibir coordenadas
            x, y, z = tvecs[i][0]
            cv2.putText(
                frame, f"ID: {ids[i][0]} XYZ: [{x:.2f}, {y:.2f}, {z:.2f}]", 
                (10, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2
            )
        
        # Desenhar contornos dos marcadores
        aruco.drawDetectedMarkers(frame, corners, ids)

    def publish_detection(self, frame, header):
        """Publicar imagem processada (com ou sem marcadores)"""
        try:
            detection_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            detection_msg.header = header
            self.detection_pub.publish(detection_msg)
        except Exception as e:
            self.get_logger().error(f'Erro ao publicar imagem: {str(e)}')

    def publish_poses(self, ids, rvecs, tvecs, header):
        """Publicar poses dos marcadores (apenas quando detectados)"""
        pose_array = PoseArray()
        pose_array.header = header
        pose_array.header.frame_id = self.camera_frame
        
        for i in range(len(ids)):
            pose = Pose()
            pose.position.x = float(tvecs[i][0][0])
            pose.position.y = float(tvecs[i][0][1])
            pose.position.z = float(tvecs[i][0][2])
            
            # Converter rotação para quaternion
            rot_mat = cv2.Rodrigues(rvecs[i])[0]
            q = self.rotation_matrix_to_quaternion(rot_mat)
            
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            
            pose_array.poses.append(pose)
        
        self.poses_pub.publish(pose_array)

    def publish_markers(self, ids, tvecs, header):
        """Publicar marcadores visuais no RViz (apenas quando detectados)"""
        marker_array = MarkerArray()
        
        for i in range(len(ids)):
            marker = Marker()
            marker.header = header
            marker.header.frame_id = self.camera_frame
            marker.ns = "aruco_markers"
            marker.id = int(ids[i][0])
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Posição e orientação
            marker.pose.position.x = float(tvecs[i][0][0])
            marker.pose.position.y = float(tvecs[i][0][1])
            marker.pose.position.z = float(tvecs[i][0][2])
            marker.scale.x = self.marker_length
            marker.scale.y = self.marker_length
            marker.scale.z = 0.01  # Marcador fino
            
            # Cor (verde semi-transparente)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            
            marker.lifetime.sec = 1  # Atualizar a cada segundo
            
            marker_array.markers.append(marker)
        
        self.markers_pub.publish(marker_array)

    def publish_transforms(self, ids, rvecs, tvecs, header):
        """Publicar transformadas TF dos marcadores (apenas quando detectados)"""
        transforms = []
        
        for i in range(len(ids)):
            t = TransformStamped()
            t.header = header
            t.header.frame_id = self.camera_frame
            t.child_frame_id = f"{self.marker_frame_prefix}{ids[i][0]}"
            
            # Posição
            t.transform.translation.x = float(tvecs[i][0][0])
            t.transform.translation.y = float(tvecs[i][0][1])
            t.transform.translation.z = float(tvecs[i][0][2])
            
            # Rotação (convertida para quaternion)
            rot_mat = cv2.Rodrigues(rvecs[i])[0]
            q = self.rotation_matrix_to_quaternion(rot_mat)
            
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            
            transforms.append(t)
        
        # Publicar todas as transformadas de uma vez
        self.tf_broadcaster.sendTransform(transforms)

    def rotation_matrix_to_quaternion(self, rot_mat):
        """Converter matriz de rotação para quaternion"""
        trace = rot_mat[0,0] + rot_mat[1,1] + rot_mat[2,2]
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (rot_mat[2,1] - rot_mat[1,2]) / S
            qy = (rot_mat[0,2] - rot_mat[2,0]) / S
            qz = (rot_mat[1,0] - rot_mat[0,1]) / S
        elif (rot_mat[0,0] > rot_mat[1,1]) and (rot_mat[0,0] > rot_mat[2,2]):
            S = np.sqrt(1.0 + rot_mat[0,0] - rot_mat[1,1] - rot_mat[2,2]) * 2
            qw = (rot_mat[2,1] - rot_mat[1,2]) / S
            qx = 0.25 * S
            qy = (rot_mat[0,1] + rot_mat[1,0]) / S
            qz = (rot_mat[0,2] + rot_mat[2,0]) / S
        elif rot_mat[1,1] > rot_mat[2,2]:
            S = np.sqrt(1.0 + rot_mat[1,1] - rot_mat[0,0] - rot_mat[2,2]) * 2
            qw = (rot_mat[0,2] - rot_mat[2,0]) / S
            qx = (rot_mat[0,1] + rot_mat[1,0]) / S
            qy = 0.25 * S
            qz = (rot_mat[1,2] + rot_mat[2,1]) / S
        else:
            S = np.sqrt(1.0 + rot_mat[2,2] - rot_mat[0,0] - rot_mat[1,1]) * 2
            qw = (rot_mat[1,0] - rot_mat[0,1]) / S
            qx = (rot_mat[0,2] + rot_mat[2,0]) / S
            qy = (rot_mat[1,2] + rot_mat[2,1]) / S
            qz = 0.25 * S
            
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    detector = ArucoDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()