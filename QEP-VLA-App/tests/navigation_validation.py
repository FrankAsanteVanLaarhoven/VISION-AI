"""
Navigation Validation Tests for QEP-VLA Application
Tests navigation engine functionality and privacy-aware path planning
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.navigation_engine import NavigationEngine, NavigationConfig, NavigationMode, Waypoint

class TestNavigationEngine(unittest.TestCase):
    """Test cases for NavigationEngine class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = NavigationConfig(
            max_speed=2.0,
            min_safety_distance=1.0,
            privacy_zone_radius=5.0,
            path_smoothing_factor=0.1,
            obstacle_detection_range=10.0,
            privacy_weight=0.3,
            efficiency_weight=0.7
        )
        self.nav_engine = NavigationEngine(self.config)
    
    def test_initialization(self):
        """Test navigation engine initialization"""
        self.assertIsNotNone(self.nav_engine)
        self.assertEqual(self.nav_engine.config.max_speed, 2.0)
        self.assertEqual(self.nav_engine.config.privacy_weight, 0.3)
        self.assertEqual(self.nav_engine.navigation_mode, NavigationMode.PRIVACY_AWARE)
        self.assertEqual(len(self.nav_engine.waypoints), 0)
        self.assertEqual(len(self.nav_engine.obstacles), 0)
        self.assertEqual(len(self.nav_engine.privacy_zones), 0)
    
    def test_set_current_position(self):
        """Test setting current position and heading"""
        position = np.array([10.0, 20.0, 5.0])
        heading = 1.57  # 90 degrees in radians
        
        self.nav_engine.set_current_position(position, heading)
        
        np.testing.assert_array_equal(self.nav_engine.current_position, position)
        self.assertEqual(self.nav_engine.current_heading, heading)
    
    def test_add_waypoint(self):
        """Test adding navigation waypoints"""
        waypoint = Waypoint(
            x=15.0,
            y=25.0,
            z=10.0,
            timestamp=datetime.now(),
            privacy_level=0.8,
            metadata={"zone_id": "test_zone"}
        )
        
        self.nav_engine.add_waypoint(waypoint)
        
        self.assertEqual(len(self.nav_engine.waypoints), 1)
        self.assertEqual(self.nav_engine.waypoints[0].x, 15.0)
        self.assertEqual(self.nav_engine.waypoints[0].privacy_level, 0.8)
    
    def test_add_obstacle(self):
        """Test adding obstacles with privacy considerations"""
        position = np.array([5.0, 5.0, 0.0])
        radius = 2.0
        privacy_impact = 0.6
        
        self.nav_engine.add_obstacle(position, radius, privacy_impact)
        
        self.assertEqual(len(self.nav_engine.obstacles), 1)
        self.assertEqual(self.nav_engine.obstacles[0]['radius'], 2.0)
        self.assertEqual(self.nav_engine.obstacles[0]['privacy_impact'], 0.6)
    
    def test_add_privacy_zone(self):
        """Test adding privacy-sensitive zones"""
        center = np.array([0.0, 0.0, 0.0])
        radius = 8.0
        privacy_level = 0.9
        
        self.nav_engine.add_privacy_zone(center, radius, privacy_level)
        
        self.assertEqual(len(self.nav_engine.privacy_zones), 1)
        self.assertEqual(self.nav_engine.privacy_zones[0]['radius'], 8.0)
        self.assertEqual(self.nav_engine.privacy_zones[0]['privacy_level'], 0.9)
    
    def test_plan_path_privacy_aware(self):
        """Test privacy-aware path planning"""
        # Set up test environment
        self.nav_engine.set_current_position(np.array([0.0, 0.0, 0.0]), 0.0)
        
        # Add privacy zone
        self.nav_engine.add_privacy_zone(np.array([5.0, 0.0, 0.0]), 3.0, 0.8)
        
        # Add obstacle
        self.nav_engine.add_obstacle(np.array([10.0, 5.0, 0.0]), 2.0, 0.4)
        
        # Plan path
        target = np.array([20.0, 0.0, 0.0])
        path = self.nav_engine.plan_path(target, NavigationMode.PRIVACY_AWARE)
        
        # Verify path
        self.assertIsInstance(path, list)
        self.assertTrue(len(path) > 0)
        self.assertTrue(all(isinstance(point, np.ndarray) for point in path))
        
        # Check that path starts at current position
        np.testing.assert_array_equal(path[0], self.nav_engine.current_position)
        
        # Check that path ends at target
        np.testing.assert_array_equal(path[-1], target)
    
    def test_plan_path_exploration(self):
        """Test exploration path planning"""
        self.nav_engine.set_current_position(np.array([0.0, 0.0, 0.0]), 0.0)
        target = np.array([15.0, 10.0, 5.0])
        
        path = self.nav_engine.plan_path(target, NavigationMode.EXPLORATION)
        
        self.assertIsInstance(path, list)
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(path[-1].tolist(), [15.0, 10.0, 5.0])
    
    def test_plan_path_direct(self):
        """Test direct path planning"""
        self.nav_engine.set_current_position(np.array([0.0, 0.0, 0.0]), 0.0)
        target = np.array([10.0, 0.0, 0.0])
        
        path = self.nav_engine.plan_path(target, NavigationMode.PATH_FOLLOWING)
        
        self.assertIsInstance(path, list)
        self.assertEqual(len(path), 2)  # Should be direct path
        self.assertEqual(path[0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(path[-1].tolist(), [10.0, 0.0, 0.0])
    
    def test_plan_path_obstacle_avoidance(self):
        """Test obstacle avoidance path planning"""
        self.nav_engine.set_current_position(np.array([0.0, 0.0, 0.0]), 0.0)
        
        # Add obstacle in direct path
        self.nav_engine.add_obstacle(np.array([5.0, 0.0, 0.0]), 2.0, 0.3)
        
        target = np.array([10.0, 0.0, 0.0])
        path = self.nav_engine.plan_path(target, NavigationMode.OBSTACLE_AVOIDANCE)
        
        self.assertIsInstance(path, list)
        self.assertTrue(len(path) > 0)
    
    def test_privacy_cost_calculation(self):
        """Test privacy cost calculation"""
        # Add privacy zones
        self.nav_engine.add_privacy_zone(np.array([0.0, 0.0, 0.0]), 5.0, 0.8)
        self.nav_engine.add_privacy_zone(np.array([10.0, 0.0, 0.0]), 3.0, 0.6)
        
        # Test positions with different privacy costs
        position1 = np.array([2.0, 0.0, 0.0])  # Inside first zone
        position2 = np.array([15.0, 0.0, 0.0])  # Inside second zone
        position3 = np.array([20.0, 0.0, 0.0])  # Outside all zones
        
        cost1 = self.nav_engine._calculate_privacy_cost(position1)
        cost2 = self.nav_engine._calculate_privacy_cost(position2)
        cost3 = self.nav_engine._calculate_privacy_cost(position3)
        
        # Position inside zones should have higher cost
        self.assertGreater(cost1, 0)
        self.assertGreater(cost2, 0)
        self.assertEqual(cost3, 0)
        
        # First zone has higher privacy level, so should have higher cost
        self.assertGreater(cost1, cost2)
    
    def test_obstacle_cost_calculation(self):
        """Test obstacle cost calculation"""
        # Add obstacles
        self.nav_engine.add_obstacle(np.array([5.0, 0.0, 0.0]), 2.0, 0.5)
        self.nav_engine.add_obstacle(np.array([15.0, 0.0, 0.0]), 1.5, 0.3)
        
        # Test positions
        position1 = np.array([5.0, 0.0, 0.0])  # Inside first obstacle
        position2 = np.array([8.0, 0.0, 0.0])  # Near first obstacle
        position3 = np.array([20.0, 0.0, 0.0])  # Far from obstacles
        
        cost1 = self.nav_engine._calculate_obstacle_cost(position1)
        cost2 = self.nav_engine._calculate_obstacle_cost(position2)
        cost3 = self.nav_engine._calculate_obstacle_cost(position3)
        
        # Position inside obstacle should have very high cost
        self.assertGreater(cost1, 1000)
        
        # Position near obstacle should have moderate cost
        self.assertGreater(cost2, 0)
        self.assertLess(cost2, cost1)
        
        # Position far from obstacles should have no cost
        self.assertEqual(cost3, 0)
    
    def test_path_smoothing(self):
        """Test path smoothing functionality"""
        # Create a simple path
        path = [
            np.array([0.0, 0.0, 0.0]),
            np.array([5.0, 2.0, 1.0]),
            np.array([10.0, 0.0, 0.0])
        ]
        
        smoothed_path = self.nav_engine._smooth_path(path)
        
        self.assertIsInstance(smoothed_path, list)
        self.assertEqual(len(smoothed_path), 3)
        
        # First and last points should remain the same
        np.testing.assert_array_equal(smoothed_path[0], path[0])
        np.testing.assert_array_equal(smoothed_path[-1], path[-1])
    
    def test_navigation_step_execution(self):
        """Test navigation step execution"""
        self.nav_engine.set_current_position(np.array([0.0, 0.0, 0.0]), 0.0)
        target = np.array([5.0, 0.0, 0.0])
        
        step_info = self.nav_engine.execute_navigation_step(target)
        
        self.assertIsInstance(step_info, dict)
        self.assertIn('timestamp', step_info)
        self.assertIn('position', step_info)
        self.assertIn('heading', step_info)
        self.assertIn('target', step_info)
        self.assertIn('speed', step_info)
        self.assertIn('privacy_cost', step_info)
        self.assertIn('distance_to_target', step_info)
        
        # Verify values
        self.assertEqual(step_info['distance_to_target'], 5.0)
        self.assertGreater(step_info['speed'], 0)
    
    def test_navigation_status(self):
        """Test navigation status retrieval"""
        # Add some test data
        self.nav_engine.add_waypoint(Waypoint(1.0, 1.0, 1.0, datetime.now(), 0.5, {}))
        self.nav_engine.add_obstacle(np.array([2.0, 2.0, 0.0]), 1.0, 0.3)
        self.nav_engine.add_privacy_zone(np.array([3.0, 3.0, 0.0]), 2.0, 0.7)
        
        status = self.nav_engine.get_navigation_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('current_position', status)
        self.assertIn('current_heading', status)
        self.assertIn('navigation_mode', status)
        self.assertIn('waypoints_count', status)
        self.assertIn('obstacles_count', status)
        self.assertIn('privacy_zones_count', status)
        self.assertIn('navigation_history_length', status)
        
        # Verify counts
        self.assertEqual(status['waypoints_count'], 1)
        self.assertEqual(status['obstacles_count'], 1)
        self.assertEqual(status['privacy_zones_count'], 1)
    
    def test_clear_navigation_history(self):
        """Test clearing navigation history"""
        # Execute some navigation steps
        self.nav_engine.set_current_position(np.array([0.0, 0.0, 0.0]), 0.0)
        self.nav_engine.execute_navigation_step(np.array([1.0, 0.0, 0.0]))
        self.nav_engine.execute_navigation_step(np.array([2.0, 0.0, 0.0]))
        
        # Verify history exists
        self.assertGreater(len(self.nav_engine.navigation_history), 0)
        
        # Clear history
        self.nav_engine.clear_navigation_history()
        
        # Verify history is cleared
        self.assertEqual(len(self.nav_engine.navigation_history), 0)
    
    def test_invalid_navigation_mode(self):
        """Test handling of invalid navigation mode"""
        self.nav_engine.set_current_position(np.array([0.0, 0.0, 0.0]), 0.0)
        target = np.array([5.0, 0.0, 0.0])
        
        # Test with invalid mode
        with self.assertRaises(ValueError):
            self.nav_engine.plan_path(target, "invalid_mode")
    
    def test_empty_path_planning(self):
        """Test path planning with no obstacles or privacy zones"""
        self.nav_engine.set_current_position(np.array([0.0, 0.0, 0.0]), 0.0)
        target = np.array([10.0, 0.0, 0.0])
        
        path = self.nav_engine.plan_path(target)
        
        self.assertIsInstance(path, list)
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(path[-1].tolist(), [10.0, 0.0, 0.0])

class TestNavigationConfig(unittest.TestCase):
    """Test cases for NavigationConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = NavigationConfig()
        
        self.assertEqual(config.max_speed, 2.0)
        self.assertEqual(config.min_safety_distance, 1.0)
        self.assertEqual(config.privacy_zone_radius, 5.0)
        self.assertEqual(config.path_smoothing_factor, 0.1)
        self.assertEqual(config.obstacle_detection_range, 10.0)
        self.assertEqual(config.privacy_weight, 0.3)
        self.assertEqual(config.efficiency_weight, 0.7)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = NavigationConfig(
            max_speed=5.0,
            min_safety_distance=2.0,
            privacy_weight=0.5,
            efficiency_weight=0.5
        )
        
        self.assertEqual(config.max_speed, 5.0)
        self.assertEqual(config.min_safety_distance, 2.0)
        self.assertEqual(config.privacy_weight, 0.5)
        self.assertEqual(config.efficiency_weight, 0.5)

class TestWaypoint(unittest.TestCase):
    """Test cases for Waypoint class"""
    
    def test_waypoint_creation(self):
        """Test waypoint creation and attributes"""
        timestamp = datetime.now()
        metadata = {"zone_id": "test", "priority": "high"}
        
        waypoint = Waypoint(
            x=10.0,
            y=20.0,
            z=5.0,
            timestamp=timestamp,
            privacy_level=0.8,
            metadata=metadata
        )
        
        self.assertEqual(waypoint.x, 10.0)
        self.assertEqual(waypoint.y, 20.0)
        self.assertEqual(waypoint.z, 5.0)
        self.assertEqual(waypoint.timestamp, timestamp)
        self.assertEqual(waypoint.privacy_level, 0.8)
        self.assertEqual(waypoint.metadata, metadata)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
