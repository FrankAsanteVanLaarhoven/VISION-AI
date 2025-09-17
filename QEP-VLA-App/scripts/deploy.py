#!/usr/bin/env python3
"""
PVLA Navigation System Deployment Script
Production-ready deployment automation for PVLA Navigation System
"""

import os
import sys
import yaml
import json
import time
import logging
import argparse
import subprocess
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import docker
import kubernetes
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings

settings = get_settings()

class PVLADeployment:
    """
    PVLA Navigation System Deployment Manager
    Handles Docker builds, Kubernetes deployments, and system validation
    """
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = project_root
        self.config_dir = self.project_root / "config"
        self.deploy_dir = self.project_root / "deploy"
        self.scripts_dir = self.project_root / "scripts"
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()  # Try in-cluster config first
        except:
            try:
                config.load_kube_config()  # Fallback to local config
            except Exception as e:
                self.logger.warning(f"Failed to load Kubernetes config: {e}")
        
        self.k8s_client = client.ApiClient()
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        
        self.logger.info(f"PVLA Deployment initialized for environment: {environment}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('deployment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def build_docker_image(self, tag: str = "pvla-navigation:latest") -> bool:
        """
        Build Docker image for PVLA Navigation System
        """
        try:
            self.logger.info(f"Building Docker image: {tag}")
            
            # Build Docker image
            image, build_logs = self.docker_client.images.build(
                path=str(self.project_root),
                tag=tag,
                dockerfile="Dockerfile",
                rm=True,
                forcerm=True
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    self.logger.info(log['stream'].strip())
                elif 'error' in log:
                    self.logger.error(log['error'])
                    return False
            
            self.logger.info(f"Successfully built Docker image: {tag}")
            return True
            
        except Exception as e:
            self.logger.error(f"Docker build failed: {e}")
            return False
    
    def push_docker_image(self, tag: str = "pvla-navigation:latest", registry: str = None) -> bool:
        """
        Push Docker image to registry
        """
        try:
            if registry:
                # Tag image for registry
                registry_tag = f"{registry}/{tag}"
                image = self.docker_client.images.get(tag)
                image.tag(registry_tag)
                tag = registry_tag
            
            self.logger.info(f"Pushing Docker image: {tag}")
            
            # Push image
            push_logs = self.docker_client.images.push(tag, stream=True, decode=True)
            
            for log in push_logs:
                if 'status' in log:
                    self.logger.info(log['status'])
                elif 'error' in log:
                    self.logger.error(log['error'])
                    return False
            
            self.logger.info(f"Successfully pushed Docker image: {tag}")
            return True
            
        except Exception as e:
            self.logger.error(f"Docker push failed: {e}")
            return False
    
    def deploy_kubernetes(self, namespace: str = "pvla-navigation") -> bool:
        """
        Deploy PVLA Navigation System to Kubernetes
        """
        try:
            self.logger.info(f"Deploying to Kubernetes namespace: {namespace}")
            
            # Load deployment configuration
            deployment_file = self.deploy_dir / "k8s" / "pvla-deployment.yaml"
            
            if not deployment_file.exists():
                self.logger.error(f"Deployment file not found: {deployment_file}")
                return False
            
            # Apply Kubernetes manifests
            result = subprocess.run([
                "kubectl", "apply", "-f", str(deployment_file)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Kubernetes deployment failed: {result.stderr}")
                return False
            
            self.logger.info("Kubernetes deployment applied successfully")
            
            # Wait for deployment to be ready
            if not self._wait_for_deployment_ready(namespace):
                return False
            
            # Validate deployment
            if not self._validate_deployment(namespace):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Kubernetes deployment failed: {e}")
            return False
    
    def _wait_for_deployment_ready(self, namespace: str, timeout: int = 300) -> bool:
        """
        Wait for deployment to be ready
        """
        try:
            self.logger.info("Waiting for deployment to be ready...")
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    deployment = self.apps_v1.read_namespaced_deployment(
                        name="pvla-navigation",
                        namespace=namespace
                    )
                    
                    if (deployment.status.ready_replicas == deployment.spec.replicas and
                        deployment.status.ready_replicas > 0):
                        self.logger.info("Deployment is ready!")
                        return True
                    
                    self.logger.info(f"Deployment status: {deployment.status.ready_replicas}/{deployment.spec.replicas} ready")
                    time.sleep(10)
                    
                except ApiException as e:
                    if e.status == 404:
                        self.logger.info("Deployment not found yet, waiting...")
                        time.sleep(5)
                    else:
                        raise
            
            self.logger.error("Deployment timeout - not ready within expected time")
            return False
            
        except Exception as e:
            self.logger.error(f"Error waiting for deployment: {e}")
            return False
    
    def _validate_deployment(self, namespace: str) -> bool:
        """
        Validate deployment health
        """
        try:
            self.logger.info("Validating deployment...")
            
            # Check pods
            pods = self.core_v1.list_namespaced_pod(namespace=namespace)
            pvla_pods = [pod for pod in pods.items if pod.metadata.labels.get('app') == 'pvla-navigation']
            
            if not pvla_pods:
                self.logger.error("No PVLA pods found")
                return False
            
            # Check pod status
            for pod in pvla_pods:
                if pod.status.phase != 'Running':
                    self.logger.error(f"Pod {pod.metadata.name} is not running: {pod.status.phase}")
                    return False
            
            # Check services
            services = self.core_v1.list_namespaced_service(namespace=namespace)
            pvla_service = None
            for service in services.items:
                if service.metadata.labels.get('app') == 'pvla-navigation':
                    pvla_service = service
                    break
            
            if not pvla_service:
                self.logger.error("PVLA service not found")
                return False
            
            # Test health endpoint
            if not self._test_health_endpoint(pvla_service):
                return False
            
            self.logger.info("Deployment validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment validation failed: {e}")
            return False
    
    def _test_health_endpoint(self, service) -> bool:
        """
        Test health endpoint
        """
        try:
            # Get service endpoint
            if service.spec.type == 'LoadBalancer':
                # Wait for external IP
                external_ip = None
                for _ in range(30):  # Wait up to 5 minutes
                    service = self.core_v1.read_namespaced_service(
                        name=service.metadata.name,
                        namespace=service.metadata.namespace
                    )
                    if service.status.load_balancer.ingress:
                        external_ip = service.status.load_balancer.ingress[0].ip
                        break
                    time.sleep(10)
                
                if not external_ip:
                    self.logger.warning("LoadBalancer external IP not available, using port-forward")
                    return self._test_with_port_forward(service)
                
                url = f"http://{external_ip}:8000/health"
            else:
                # Use port-forward for testing
                return self._test_with_port_forward(service)
            
            # Test health endpoint
            import requests
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'healthy':
                    self.logger.info("Health endpoint test successful")
                    return True
                else:
                    self.logger.error(f"Health check failed: {health_data}")
                    return False
            else:
                self.logger.error(f"Health endpoint returned status {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Health endpoint test failed: {e}")
            return False
    
    def _test_with_port_forward(self, service) -> bool:
        """
        Test health endpoint using port-forward
        """
        try:
            # Start port-forward in background
            port_forward_process = subprocess.Popen([
                "kubectl", "port-forward",
                f"service/{service.metadata.name}",
                "8000:8000",
                f"-n", service.metadata.namespace
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for port-forward to be ready
            time.sleep(5)
            
            # Test health endpoint
            import requests
            response = requests.get("http://localhost:8000/health", timeout=30)
            
            # Stop port-forward
            port_forward_process.terminate()
            port_forward_process.wait()
            
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'healthy':
                    self.logger.info("Health endpoint test successful (port-forward)")
                    return True
                else:
                    self.logger.error(f"Health check failed: {health_data}")
                    return False
            else:
                self.logger.error(f"Health endpoint returned status {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Port-forward health test failed: {e}")
            return False
    
    def run_system_tests(self, namespace: str = "pvla-navigation") -> bool:
        """
        Run comprehensive system tests
        """
        try:
            self.logger.info("Running system tests...")
            
            # Test navigation endpoint
            if not self._test_navigation_endpoint():
                return False
            
            # Test privacy monitoring
            if not self._test_privacy_monitoring():
                return False
            
            # Test quantum infrastructure
            if not self._test_quantum_infrastructure():
                return False
            
            # Test performance benchmarks
            if not self._test_performance_benchmarks():
                return False
            
            self.logger.info("All system tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"System tests failed: {e}")
            return False
    
    def _test_navigation_endpoint(self) -> bool:
        """
        Test navigation endpoint
        """
        try:
            self.logger.info("Testing navigation endpoint...")
            
            # Create test navigation request
            test_request = {
                "camera_frame": {
                    "frame_data": [[[255, 255, 255] for _ in range(224)] for _ in range(224)],
                    "width": 224,
                    "height": 224
                },
                "language_command": {
                    "command": "Move forward carefully"
                },
                "navigation_context": {
                    "current_position": [0.0, 0.0, 0.0],
                    "current_orientation": [0.0, 0.0, 0.0],
                    "target_position": [1.0, 0.0, 0.0],
                    "environment_data": {},
                    "safety_constraints": {},
                    "objectives": ["move_forward"]
                }
            }
            
            # Test navigation request
            import requests
            response = requests.post(
                "http://localhost:8000/navigate",
                json=test_request,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('navigation_action') is not None:
                    self.logger.info("Navigation endpoint test successful")
                    return True
                else:
                    self.logger.error("Navigation endpoint returned invalid response")
                    return False
            else:
                self.logger.error(f"Navigation endpoint returned status {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Navigation endpoint test failed: {e}")
            return False
    
    def _test_privacy_monitoring(self) -> bool:
        """
        Test privacy monitoring functionality
        """
        try:
            self.logger.info("Testing privacy monitoring...")
            
            # Test privacy status endpoint
            import requests
            response = requests.get("http://localhost:8000/status", timeout=30)
            
            if response.status_code == 200:
                status = response.json()
                privacy_metadata = status.get('component_health', {}).get('privacy', {})
                
                if privacy_metadata:
                    self.logger.info("Privacy monitoring test successful")
                    return True
                else:
                    self.logger.error("Privacy monitoring not found in status")
                    return False
            else:
                self.logger.error(f"Privacy monitoring test failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Privacy monitoring test failed: {e}")
            return False
    
    def _test_quantum_infrastructure(self) -> bool:
        """
        Test quantum infrastructure
        """
        try:
            self.logger.info("Testing quantum infrastructure...")
            
            # Test quantum status
            import requests
            response = requests.get("http://localhost:8000/status", timeout=30)
            
            if response.status_code == 200:
                status = response.json()
                quantum_health = status.get('component_health', {}).get('meta_learning', {})
                
                if quantum_health and quantum_health.get('status') == 'healthy':
                    self.logger.info("Quantum infrastructure test successful")
                    return True
                else:
                    self.logger.error("Quantum infrastructure not healthy")
                    return False
            else:
                self.logger.error(f"Quantum infrastructure test failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Quantum infrastructure test failed: {e}")
            return False
    
    def _test_performance_benchmarks(self) -> bool:
        """
        Test performance benchmarks
        """
        try:
            self.logger.info("Testing performance benchmarks...")
            
            # Test multiple navigation requests
            import requests
            import time
            
            start_time = time.time()
            successful_requests = 0
            total_requests = 10
            
            for i in range(total_requests):
                test_request = {
                    "camera_frame": {
                        "frame_data": [[[255, 255, 255] for _ in range(224)] for _ in range(224)],
                        "width": 224,
                        "height": 224
                    },
                    "language_command": {
                        "command": f"Test command {i}"
                    },
                    "navigation_context": {
                        "current_position": [0.0, 0.0, 0.0],
                        "current_orientation": [0.0, 0.0, 0.0],
                        "target_position": [1.0, 0.0, 0.0],
                        "environment_data": {},
                        "safety_constraints": {},
                        "objectives": ["move_forward"]
                    }
                }
                
                response = requests.post(
                    "http://localhost:8000/navigate",
                    json=test_request,
                    timeout=30
                )
                
                if response.status_code == 200:
                    successful_requests += 1
                
                time.sleep(0.1)  # Small delay between requests
            
            total_time = time.time() - start_time
            avg_response_time = total_time / total_requests
            
            if successful_requests == total_requests and avg_response_time < 1.0:
                self.logger.info(f"Performance benchmark test successful: {avg_response_time:.2f}s avg response time")
                return True
            else:
                self.logger.error(f"Performance benchmark test failed: {successful_requests}/{total_requests} successful, {avg_response_time:.2f}s avg")
                return False
                
        except Exception as e:
            self.logger.error(f"Performance benchmark test failed: {e}")
            return False
    
    def cleanup_deployment(self, namespace: str = "pvla-navigation") -> bool:
        """
        Cleanup deployment
        """
        try:
            self.logger.info(f"Cleaning up deployment in namespace: {namespace}")
            
            # Delete deployment
            deployment_file = self.deploy_dir / "k8s" / "pvla-deployment.yaml"
            
            result = subprocess.run([
                "kubectl", "delete", "-f", str(deployment_file)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.warning(f"Cleanup warning: {result.stderr}")
            
            self.logger.info("Deployment cleanup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment cleanup failed: {e}")
            return False

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="PVLA Navigation System Deployment")
    parser.add_argument("--environment", default="production", help="Deployment environment")
    parser.add_argument("--action", required=True, choices=["build", "deploy", "test", "cleanup", "full"], help="Deployment action")
    parser.add_argument("--tag", default="pvla-navigation:latest", help="Docker image tag")
    parser.add_argument("--registry", help="Docker registry URL")
    parser.add_argument("--namespace", default="pvla-navigation", help="Kubernetes namespace")
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    deployment = PVLADeployment(args.environment)
    
    success = True
    
    if args.action == "build":
        success = deployment.build_docker_image(args.tag)
        if success and args.registry:
            success = deployment.push_docker_image(args.tag, args.registry)
    
    elif args.action == "deploy":
        success = deployment.deploy_kubernetes(args.namespace)
    
    elif args.action == "test":
        success = deployment.run_system_tests(args.namespace)
    
    elif args.action == "cleanup":
        success = deployment.cleanup_deployment(args.namespace)
    
    elif args.action == "full":
        # Full deployment pipeline
        success = deployment.build_docker_image(args.tag)
        if success and args.registry:
            success = deployment.push_docker_image(args.tag, args.registry)
        if success:
            success = deployment.deploy_kubernetes(args.namespace)
        if success:
            success = deployment.run_system_tests(args.namespace)
    
    if success:
        print("✅ Deployment completed successfully!")
        sys.exit(0)
    else:
        print("❌ Deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()