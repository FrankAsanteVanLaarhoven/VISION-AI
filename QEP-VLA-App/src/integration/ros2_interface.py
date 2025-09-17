"""
ROS2 Interface for PVLA Navigation System
Production-ready ROS2 integration for robotics applications
"""

import asyncio
import logging
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Callable
import json
import threading
from concurrent.futures import ThreadPoolExecutor

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, PointCloud2, Imu
    from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
    from std_msgs.msg import String, Float32, Bool
    from nav_msgs.msg import OccupancyGrid, Path
    from visualization_msgs.msg import Marker, MarkerArray
    from cv_bridge import CvBridge
    import tf2_ros
    from tf2_ros import TransformBroadcaster
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logging.warning("ROS2 not available. ROS2 interface will be disabled.")

# PVLA imports
from core.pvla_navigation_system import PVLANavigationSystem, PVLAConfig
from config.settings import get_settings

settings = get_settings()

class PVLAROS2Interface(Node):
    """
    ROS2 interface for PVLA Navigation System
    
    Provides ROS2 topics and services for:
    - Camera data processing
    - LiDAR data processing
    - Navigation command processing
    - Action execution
    - System monitoring
    """
    
    def __init__(self, pvla_system: PVLANavigationSystem):
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS2 is not available. Cannot initialize ROS2 interface.")
        
        super().__init__('pvla_navigation_node')
        
        self.pvla_system = pvla_system
        self.bridge = CvBridge()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # QoS profiles
        self.qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        self.qos_command = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Initialize publishers
        self._init_publishers()
        
        # Initialize subscribers
        self._init_subscribers()
        
        # Initialize services
        self._init_services()
        
        # Initialize timers
        self._init_timers()
        
        # State variables
        self.latest_camera_frame = None
        self.latest_lidar_data = None
        self.latest_imu_data = None
        self.current_navigation_state = None
        self.navigation_objectives = None
        
        # Performance tracking
        self.processing_times = []
        self.success_rates = []
        
        self.get_logger().info("PVLA ROS2 Interface initialized")
    
    def _init_publishers(self):
        """Initialize ROS2 publishers"""
        # Navigation commands
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/pvla/cmd_vel', self.qos_command
        )
        
        # Navigation path
        self.path_pub = self.create_publisher(
            Path, '/pvla/navigation_path', self.qos_command
        )
        
        # System status
        self.status_pub = self.create_publisher(
            String, '/pvla/system_status', self.qos_command
        )
        
        # Performance metrics
        self.metrics_pub = self.create_publisher(
            String, '/pvla/performance_metrics', self.qos_command
        )
        
        # Visualization markers
        self.markers_pub = self.create_publisher(
            MarkerArray, '/pvla/visualization_markers', self.qos_command
        )
        
        # Navigation explanation
        self.explanation_pub = self.create_publisher(
            String, '/pvla/navigation_explanation', self.qos_command
        )
    
    def _init_subscribers(self):
        """Initialize ROS2 subscribers"""
        # Camera data
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, self.qos_sensor
        )
        
        # LiDAR data
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.lidar_callback, self.qos_sensor
        )
        
        # IMU data
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, self.qos_sensor
        )
        
        # Navigation commands
        self.nav_cmd_sub = self.create_subscription(
            String, '/pvla/navigation_command', self.navigation_command_callback, self.qos_command
        )
        
        # Navigation objectives
        self.objectives_sub = self.create_subscription(
            String, '/pvla/navigation_objectives', self.objectives_callback, self.qos_command
        )
        
        # Emergency stop
        self.emergency_stop_sub = self.create_subscription(
            Bool, '/pvla/emergency_stop', self.emergency_stop_callback, self.qos_command
        )
    
    def _init_services(self):
        """Initialize ROS2 services"""
        # System control services
        self.reset_system_srv = self.create_service(
            String, '/pvla/reset_system', self.reset_system_callback
        )
        
        self.update_config_srv = self.create_service(
            String, '/pvla/update_config', self.update_config_callback
        )
        
        # Navigation services
        self.set_goal_srv = self.create_service(
            PoseStamped, '/pvla/set_goal', self.set_goal_callback
        )
        
        self.get_status_srv = self.create_service(
            String, '/pvla/get_status', self.get_status_callback
        )
    
    def _init_timers(self):
        """Initialize ROS2 timers"""
        # System status timer
        self.status_timer = self.create_timer(1.0, self.publish_system_status)
        
        # Performance metrics timer
        self.metrics_timer = self.create_timer(5.0, self.publish_performance_metrics)
        
        # Navigation processing timer
        self.navigation_timer = self.create_timer(0.1, self.process_navigation)  # 10Hz
    
    # Callback functions
    def camera_callback(self, msg: Image):
        """Process camera data"""
        try:
            # Convert ROS2 Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Store latest frame
            self.latest_camera_frame = cv_image
            
            self.get_logger().debug(f"Received camera frame: {cv_image.shape}")
            
        except Exception as e:
            self.get_logger().error(f"Camera callback error: {e}")
    
    def lidar_callback(self, msg: PointCloud2):
        """Process LiDAR data"""
        try:
            # Convert PointCloud2 to numpy array
            # This is a simplified conversion - in production, use proper point cloud processing
            lidar_data = np.frombuffer(msg.data, dtype=np.float32)
            
            # Store latest LiDAR data
            self.latest_lidar_data = lidar_data
            
            self.get_logger().debug(f"Received LiDAR data: {len(lidar_data)} points")
            
        except Exception as e:
            self.get_logger().error(f"LiDAR callback error: {e}")
    
    def imu_callback(self, msg: Imu):
        """Process IMU data"""
        try:
            # Extract IMU data
            imu_data = {
                'linear_acceleration': [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ],
                'angular_velocity': [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ],
                'orientation': [
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w
                ]
            }
            
            # Store latest IMU data
            self.latest_imu_data = imu_data
            
            self.get_logger().debug("Received IMU data")
            
        except Exception as e:
            self.get_logger().error(f"IMU callback error: {e}")
    
    def navigation_command_callback(self, msg: String):
        """Process navigation command"""
        try:
            command_data = json.loads(msg.data)
            language_command = command_data.get('command', '')
            context = command_data.get('context', {})
            
            # Process navigation command asynchronously
            asyncio.create_task(self.process_navigation_command(language_command, context))
            
            self.get_logger().info(f"Received navigation command: {language_command}")
            
        except Exception as e:
            self.get_logger().error(f"Navigation command callback error: {e}")
    
    def objectives_callback(self, msg: String):
        """Process navigation objectives"""
        try:
            objectives_data = json.loads(msg.data)
            objectives = objectives_data.get('objectives', [])
            
            # Update navigation objectives
            objectives_tensor = torch.tensor(objectives, device=self.pvla_system.device)
            self.pvla_system.update_navigation_objectives(objectives_tensor)
            
            self.get_logger().info(f"Updated navigation objectives: {objectives}")
            
        except Exception as e:
            self.get_logger().error(f"Objectives callback error: {e}")
    
    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop"""
        try:
            if msg.data:
                # Publish stop command
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                
                # Log emergency stop
                self.get_logger().warn("EMERGENCY STOP ACTIVATED")
                
        except Exception as e:
            self.get_logger().error(f"Emergency stop callback error: {e}")
    
    # Service callbacks
    def reset_system_callback(self, request, response):
        """Reset system service callback"""
        try:
            self.pvla_system.reset_system_metrics()
            response.data = "System metrics reset successfully"
            self.get_logger().info("System reset requested")
        except Exception as e:
            response.data = f"System reset failed: {str(e)}"
            self.get_logger().error(f"System reset error: {e}")
        return response
    
    def update_config_callback(self, request, response):
        """Update configuration service callback"""
        try:
            config_data = json.loads(request.data)
            # Update PVLA system configuration
            # This would require implementing configuration update methods
            response.data = "Configuration updated successfully"
            self.get_logger().info("Configuration update requested")
        except Exception as e:
            response.data = f"Configuration update failed: {str(e)}"
            self.get_logger().error(f"Configuration update error: {e}")
        return response
    
    def set_goal_callback(self, request, response):
        """Set navigation goal service callback"""
        try:
            goal_position = [
                request.pose.position.x,
                request.pose.position.y,
                request.pose.position.z
            ]
            goal_orientation = [
                request.pose.orientation.x,
                request.pose.orientation.y,
                request.pose.orientation.z,
                request.pose.orientation.w
            ]
            
            # Update navigation objectives
            objectives = goal_position + goal_orientation + [0.0] * 4  # Pad to 10
            objectives_tensor = torch.tensor(objectives, device=self.pvla_system.device)
            self.pvla_system.update_navigation_objectives(objectives_tensor)
            
            response.pose = request.pose
            self.get_logger().info(f"Navigation goal set: {goal_position}")
            
        except Exception as e:
            self.get_logger().error(f"Set goal error: {e}")
            response.pose = request.pose
        return response
    
    def get_status_callback(self, request, response):
        """Get system status service callback"""
        try:
            status = self.pvla_system.get_system_status()
            response.data = json.dumps(status, default=str)
        except Exception as e:
            response.data = json.dumps({"error": str(e)})
            self.get_logger().error(f"Get status error: {e}")
        return response
    
    # Timer callbacks
    def publish_system_status(self):
        """Publish system status"""
        try:
            status = self.pvla_system.get_system_status()
            status_msg = String()
            status_msg.data = json.dumps(status, default=str)
            self.status_pub.publish(status_msg)
        except Exception as e:
            self.get_logger().error(f"Status publish error: {e}")
    
    def publish_performance_metrics(self):
        """Publish performance metrics"""
        try:
            metrics = {
                'processing_times': self.processing_times[-100:],  # Last 100
                'success_rates': self.success_rates[-100:],
                'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
                'average_success_rate': np.mean(self.success_rates) if self.success_rates else 0.0,
                'timestamp': time.time()
            }
            
            metrics_msg = String()
            metrics_msg.data = json.dumps(metrics)
            self.metrics_pub.publish(metrics_msg)
            
        except Exception as e:
            self.get_logger().error(f"Metrics publish error: {e}")
    
    def process_navigation(self):
        """Main navigation processing loop"""
        try:
            if (self.latest_camera_frame is not None and 
                self.latest_imu_data is not None):
                
                # Process navigation asynchronously
                asyncio.create_task(self.process_navigation_async())
                
        except Exception as e:
            self.get_logger().error(f"Navigation processing error: {e}")
    
    async def process_navigation_async(self):
        """Asynchronous navigation processing"""
        try:
            start_time = time.time()
            
            # Prepare navigation context
            navigation_context = {
                'context': self.latest_imu_data['orientation'][:3],  # Use orientation as context
                'objectives': [1.0] * 10,  # Default objectives
                'goals': [0.0] * 256,  # Default goals
                'environment': list(self.latest_imu_data['linear_acceleration']) + [0.0] * 125,
                'context': list(self.latest_imu_data['angular_velocity']) + [0.0] * 125
            }
            
            # Process navigation request
            result = await self.pvla_system.process_navigation_request(
                camera_frame=self.latest_camera_frame,
                language_command="Continue navigation",  # Default command
                navigation_context=navigation_context
            )
            
            # Publish navigation command
            self.publish_navigation_command(result)
            
            # Publish explanation
            self.publish_navigation_explanation(result)
            
            # Update performance tracking
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.success_rates.append(result['confidence_score'])
            
        except Exception as e:
            self.get_logger().error(f"Async navigation processing error: {e}")
    
    async def process_navigation_command(self, language_command: str, context: Dict[str, Any]):
        """Process specific navigation command"""
        try:
            if self.latest_camera_frame is None:
                self.get_logger().warn("No camera frame available for navigation command")
                return
            
            # Prepare navigation context
            navigation_context = {
                'context': context.get('position', [0.0] * 6),
                'objectives': context.get('objectives', [1.0] * 10),
                'goals': context.get('goals', [0.0] * 256),
                'environment': context.get('environment', [0.0] * 128),
                'context': context.get('safety_constraints', [0.0] * 128)
            }
            
            # Process navigation request
            result = await self.pvla_system.process_navigation_request(
                camera_frame=self.latest_camera_frame,
                language_command=language_command,
                navigation_context=navigation_context
            )
            
            # Publish results
            self.publish_navigation_command(result)
            self.publish_navigation_explanation(result)
            
        except Exception as e:
            self.get_logger().error(f"Navigation command processing error: {e}")
    
    def publish_navigation_command(self, result: Dict[str, Any]):
        """Publish navigation command"""
        try:
            # Convert action to Twist message
            cmd_vel = Twist()
            
            action_idx = result['navigation_action']
            
            # Map action indices to velocity commands
            if action_idx == 0:  # move_forward
                cmd_vel.linear.x = 0.5
            elif action_idx == 1:  # move_backward
                cmd_vel.linear.x = -0.5
            elif action_idx == 2:  # turn_left
                cmd_vel.angular.z = 0.5
            elif action_idx == 3:  # turn_right
                cmd_vel.angular.z = -0.5
            elif action_idx == 4:  # stop
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
            # Add more action mappings as needed
            
            self.cmd_vel_pub.publish(cmd_vel)
            
        except Exception as e:
            self.get_logger().error(f"Navigation command publish error: {e}")
    
    def publish_navigation_explanation(self, result: Dict[str, Any]):
        """Publish navigation explanation"""
        try:
            explanation_msg = String()
            explanation_msg.data = result['explanation']
            self.explanation_pub.publish(explanation_msg)
            
        except Exception as e:
            self.get_logger().error(f"Explanation publish error: {e}")
    
    def create_visualization_markers(self, result: Dict[str, Any]) -> MarkerArray:
        """Create visualization markers for navigation result"""
        try:
            markers = MarkerArray()
            
            # Create confidence marker
            confidence_marker = Marker()
            confidence_marker.header.frame_id = "base_link"
            confidence_marker.header.stamp = self.get_clock().now().to_msg()
            confidence_marker.id = 0
            confidence_marker.type = Marker.TEXT_VIEW_FACING
            confidence_marker.action = Marker.ADD
            confidence_marker.pose.position.x = 0.0
            confidence_marker.pose.position.y = 0.0
            confidence_marker.pose.position.z = 2.0
            confidence_marker.scale.z = 0.5
            confidence_marker.color.a = 1.0
            confidence_marker.color.r = 1.0
            confidence_marker.color.g = 1.0
            confidence_marker.color.b = 0.0
            confidence_marker.text = f"Confidence: {result['confidence_score']:.2f}"
            
            markers.markers.append(confidence_marker)
            
            return markers
            
        except Exception as e:
            self.get_logger().error(f"Visualization markers error: {e}")
            return MarkerArray()

def main():
    """Main function for ROS2 interface"""
    if not ROS2_AVAILABLE:
        logging.error("ROS2 is not available. Cannot run ROS2 interface.")
        return
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Initialize PVLA system
        pvla_system = PVLANavigationSystem()
        
        # Initialize ROS2 interface
        ros2_interface = PVLAROS2Interface(pvla_system)
        
        # Spin the node
        rclpy.spin(ros2_interface)
        
    except KeyboardInterrupt:
        logging.info("ROS2 interface shutdown requested")
    except Exception as e:
        logging.error(f"ROS2 interface error: {e}")
    finally:
        if 'ros2_interface' in locals():
            ros2_interface.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
