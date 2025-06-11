#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco #type: ignore
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
        self.declare_parameter('position_threshold', 0.02)  # 2cm - CONFIGURAÇÃO DO FILTRO
        self.declare_parameter('angle_threshold', 0.15)  # ~8.6 graus - CONFIGURAÇÃO DO FILTRO
        
        # Parâmetros da câmera
        self.declare_parameter('camera_matrix', [1000.0, 0.0, 320.0, 
                                               0.0, 1000.0, 240.0, 
                                               0.0, 0.0, 1.0])
        self.declare_parameter('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Obter parâmetros
        self.marker_length = self.get_parameter('marker_length').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.marker_frame_prefix = self.get_parameter('marker_frame_prefix').value
        self.pos_threshold = self.get_parameter('position_threshold').value
        self.ang_threshold = self.get_parameter('angle_threshold').value
        
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
        
        # Armazenar últimas poses válidas
        self.last_valid_poses = {}
        
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
        
        self.get_logger().info("Aruco Detector inicializado com filtro de faixa e eixos visíveis")

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
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Erro ao converter imagem: {str(e)}')
            return
            
        # Criar cópia da imagem original para desenhar os marcadores
        processed_frame = frame.copy()
        
        # Processar frame para detecção de ArUco
        markers_detected = self.process_frame(processed_frame)
        
        # Publicar imagem processada (com eixos visíveis)
        self.publish_detection(processed_frame, msg.header)
        
        if markers_detected:
            corners, ids, rvecs, tvecs = markers_detected
            self.handle_detected_markers(ids, rvecs, tvecs, msg.header)

    def handle_detected_markers(self, ids, rvecs, tvecs, header):
        """Processar marcadores detectados com filtro de faixa"""
        pose_array = PoseArray()
        pose_array.header = header
        pose_array.header.frame_id = self.camera_frame
        
        marker_array = MarkerArray()
        
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            current_pos = tvecs[i][0]
            current_rot = rvecs[i][0]
            
            # Verificar se devemos atualizar a pose
            if self.should_update_pose(marker_id, current_pos, current_rot):
                self.last_valid_poses[marker_id] = {
                    'position': current_pos,
                    'rotation': current_rot
                }
            
            # Se temos uma pose válida para este marcador
            if marker_id in self.last_valid_poses:
                # Adicionar à PoseArray
                pose = self.create_pose(marker_id)
                pose_array.poses.append(pose)
                
                # Adicionar ao MarkerArray
                marker = self.create_marker(marker_id, header)
                marker_array.markers.append(marker)
        
        # Publicar todas as mensagens
        self.poses_pub.publish(pose_array)
        self.markers_pub.publish(marker_array)
        
        # Publicar transformadas TF
        self.publish_transforms(header)

    def should_update_pose(self, marker_id, current_pos, current_rot):
        """Determinar se a pose deve ser atualizada baseado nos limiares"""
        if marker_id not in self.last_valid_poses:
            return True  # Primeira detecção deste marcador
            
        last_pos = self.last_valid_poses[marker_id]['position']
        last_rot = self.last_valid_poses[marker_id]['rotation']
        
        # Calcular diferenças
        pos_diff = np.linalg.norm(current_pos - last_pos)
        rot_diff = np.linalg.norm(current_rot - last_rot)
        
        # Atualizar apenas se exceder algum limiar
        return (pos_diff > self.pos_threshold) or (rot_diff > self.ang_threshold)

    def create_pose(self, marker_id):
        """Criar mensagem Pose a partir da última pose válida"""
        last_pose = self.last_valid_poses[marker_id]
        pose = Pose()
        
        # Posição
        pose.position.x = float(last_pose['position'][0])
        pose.position.y = float(last_pose['position'][1])
        pose.position.z = float(last_pose['position'][2])
        
        # Orientação (convertida para quaternion)
        rot_mat = cv2.Rodrigues(last_pose['rotation'])[0]
        q = self.rotation_matrix_to_quaternion(rot_mat)
        
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        
        return pose

    def create_marker(self, marker_id, header):
        """Criar marcador visual para RViz"""
        last_pose = self.last_valid_poses[marker_id]
        marker = Marker()
        
        marker.header = header
        marker.header.frame_id = self.camera_frame
        marker.ns = "aruco_markers"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Posição
        marker.pose.position.x = float(last_pose['position'][0])
        marker.pose.position.y = float(last_pose['position'][1])
        marker.pose.position.z = float(last_pose['position'][2])
        
        # Orientação
        rot_mat = cv2.Rodrigues(last_pose['rotation'])[0]
        q = self.rotation_matrix_to_quaternion(rot_mat)
        
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
        
        # Tamanho
        marker.scale.x = self.marker_length
        marker.scale.y = self.marker_length
        marker.scale.z = 0.01  # Marcador fino
        
        # Cor (verde semi-transparente)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.7
        
        marker.lifetime.sec = 1  # Atualizar a cada segundo
        
        return marker

    def publish_transforms(self, header):
        """Publicar transformadas TF para todos os marcadores ativos"""
        transforms = []
        
        for marker_id, pose_data in self.last_valid_poses.items():
            t = TransformStamped()
            t.header = header
            t.header.frame_id = self.camera_frame
            t.child_frame_id = f"{self.marker_frame_prefix}{marker_id}"
            
            # Posição
            t.transform.translation.x = float(pose_data['position'][0])
            t.transform.translation.y = float(pose_data['position'][1])
            t.transform.translation.z = float(pose_data['position'][2])
            
            # Rotação (convertida para quaternion)
            rot_mat = cv2.Rodrigues(pose_data['rotation'])[0]
            q = self.rotation_matrix_to_quaternion(rot_mat)
            
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            
            transforms.append(t)
        
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def process_frame(self, frame):
        """Processar frame para detecção de ArUco"""
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
            
            # Desenhar os marcadores e eixos de pose
            for i in range(len(ids)):
                # Desenhar eixos de referência
                cv2.drawFrameAxes(
                    frame, self.camera_matrix, self.dist_coeffs,
                    rvecs[i], tvecs[i], self.marker_length
                )
            
            # Desenhar contornos dos marcadores
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            return corners, ids, rvecs, tvecs
        
        return None

    def publish_detection(self, frame, header):
        """Publicar imagem processada"""
        try:
            detection_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            detection_msg.header = header
            self.detection_pub.publish(detection_msg)
        except Exception as e:
            self.get_logger().error(f'Erro ao publicar imagem: {str(e)}')

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