#!/usr/bin/env python3
"""
Performance Validation Suite for QEP-VLA Platform
Validates system performance, latency, and scalability requirements
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    latency_ms: float
    throughput_rps: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None

@dataclass
class ValidationResult:
    """Validation test result"""
    test_name: str
    passed: bool
    metrics: PerformanceMetrics
    details: str
    timestamp: str

class PerformanceValidator:
    """Performance validation suite for QEP-VLA Platform"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[ValidationResult] = []
        
    def generate_synthetic_data(self, size: int = 1000) -> Dict:
        """Generate synthetic test data"""
        return {
            "navigation_requests": [
                {
                    "start_point": [np.random.uniform(0, 100), np.random.uniform(0, 100)],
                    "end_point": [np.random.uniform(0, 100), np.random.uniform(0, 100)],
                    "privacy_level": np.random.choice(["low", "medium", "high"]),
                    "sensor_data": {
                        "camera": np.random.rand(224, 224, 3).tolist(),
                        "lidar": np.random.rand(1000, 3).tolist()
                    }
                }
                for _ in range(size)
            ],
            "privacy_transforms": [
                {
                    "data": np.random.rand(100, 100).tolist(),
                    "transform_type": np.random.choice([
                        "QUANTUM_NOISE", "ENTANGLEMENT_MASKING", 
                        "SUPERPOSITION_ENCODING", "QUANTUM_KEY_ENCRYPTION"
                    ]),
                    "privacy_budget": np.random.uniform(0.1, 1.0)
                }
                for _ in range(size)
            ]
        }
        
    def test_latency_requirements(self, target_latency_ms: float = 50.0) -> ValidationResult:
        """Test if system meets latency requirements"""
        logger.info(f"Testing latency requirements (target: {target_latency_ms}ms)")
        
        test_data = self.generate_synthetic_data(size=100)
        latencies = []
        
        for request in test_data["navigation_requests"][:50]:  # Test with 50 requests
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/navigate",
                    json=request,
                    timeout=10
                )
                if response.status_code == 200:
                    latency_ms = (time.time() - start_time) * 1000
                    latencies.append(latency_ms)
            except Exception as e:
                logger.warning(f"Request failed: {e}")
                
        if not latencies:
            return ValidationResult(
                test_name="Latency Requirements",
                passed=False,
                metrics=PerformanceMetrics(0, 0, 100.0, 0, 0),
                details="No successful requests to measure latency",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        passed = p95_latency <= target_latency_ms
        
        metrics = PerformanceMetrics(
            latency_ms=avg_latency,
            throughput_rps=len(latencies) / (max(latencies) / 1000),
            error_rate=0.0,
            cpu_usage=0.0,
            memory_usage=0.0
        )
        
        details = f"P95: {p95_latency:.2f}ms, P99: {p99_latency:.2f}ms, Target: {target_latency_ms}ms"
        
        return ValidationResult(
            test_name="Latency Requirements",
            passed=passed,
            metrics=metrics,
            details=details,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    def test_load_capacity(self, concurrent_requests: int = 100) -> ValidationResult:
        """Test system load capacity with concurrent requests"""
        logger.info(f"Testing load capacity with {concurrent_requests} concurrent requests")
        
        test_data = self.generate_synthetic_data(size=concurrent_requests)
        start_time = time.time()
        
        async def make_request(request_data):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: requests.post(
                        f"{self.base_url}/navigate",
                        json=request_data,
                        timeout=30
                    )
                )
                return response.status_code == 200
            except Exception:
                return False
                
        # Create concurrent tasks
        tasks = [
            make_request(request) 
            for request in test_data["navigation_requests"]
        ]
        
        # Execute concurrently
        results = asyncio.run(asyncio.gather(*tasks))
        
        end_time = time.time()
        duration = end_time - start_time
        
        successful_requests = sum(results)
        error_rate = (concurrent_requests - successful_requests) / concurrent_requests * 100
        throughput = successful_requests / duration
        
        passed = error_rate <= 5.0  # Allow up to 5% error rate
        
        metrics = PerformanceMetrics(
            latency_ms=duration * 1000 / concurrent_requests,
            throughput_rps=throughput,
            error_rate=error_rate,
            cpu_usage=0.0,
            memory_usage=0.0
        )
        
        details = f"Success: {successful_requests}/{concurrent_requests}, Error Rate: {error_rate:.1f}%, Throughput: {throughput:.1f} RPS"
        
        return ValidationResult(
            test_name="Load Capacity",
            passed=passed,
            metrics=metrics,
            details=details,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    def test_privacy_guarantees(self) -> ValidationResult:
        """Test privacy transformation guarantees"""
        logger.info("Testing privacy transformation guarantees")
        
        test_data = self.generate_synthetic_data(size=50)
        privacy_results = []
        
        for transform in test_data["privacy_transforms"]:
            try:
                response = requests.post(
                    f"{self.base_url}/privacy/transform",
                    json=transform,
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    privacy_results.append(result)
            except Exception as e:
                logger.warning(f"Privacy transform request failed: {e}")
                
        if not privacy_results:
            return ValidationResult(
                test_name="Privacy Guarantees",
                passed=False,
                metrics=PerformanceMetrics(0, 0, 100.0, 0, 0),
                details="No successful privacy transformations",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
        # Check privacy metrics
        privacy_scores = [result.get("privacy_score", 0) for result in privacy_results]
        avg_privacy_score = statistics.mean(privacy_scores)
        
        # Privacy score should be above 0.8 for high privacy
        passed = avg_privacy_score >= 0.8
        
        metrics = PerformanceMetrics(
            latency_ms=0.0,
            throughput_rps=len(privacy_results),
            error_rate=0.0,
            cpu_usage=0.0,
            memory_usage=0.0
        )
        
        details = f"Average Privacy Score: {avg_privacy_score:.3f}, Target: >=0.8"
        
        return ValidationResult(
            test_name="Privacy Guarantees",
            passed=passed,
            metrics=metrics,
            details=details,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    def run_full_validation(self) -> Dict:
        """Run complete performance validation suite"""
        logger.info("Starting full performance validation suite")
        
        # Run all tests
        self.results = [
            self.test_latency_requirements(),
            self.test_load_capacity(),
            self.test_privacy_guarantees()
        ]
        
        # Calculate overall metrics
        overall_passed = all(result.passed for result in self.results)
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.passed)
        
        # Aggregate metrics
        avg_latency = statistics.mean([
            result.metrics.latency_ms for result in self.results 
            if result.metrics.latency_ms > 0
        ]) if any(result.metrics.latency_ms > 0 for result in self.results) else 0
        
        avg_throughput = statistics.mean([
            result.metrics.throughput_rps for result in self.results 
            if result.metrics.throughput_rps > 0
        ]) if any(result.metrics.throughput_rps > 0 for result in self.results) else 0
        
        avg_error_rate = statistics.mean([
            result.metrics.error_rate for result in self.results
        ])
        
        # Generate comprehensive report
        report = {
            "validation_summary": {
                "overall_status": "PASSED" if overall_passed else "FAILED",
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests) * 100,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "performance_metrics": {
                "average_latency_ms": round(avg_latency, 2),
                "average_throughput_rps": round(avg_throughput, 2),
                "average_error_rate_percent": round(avg_error_rate, 2)
            },
            "test_results": [
                {
                    "test_name": result.test_name,
                    "status": "PASSED" if result.passed else "FAILED",
                    "metrics": {
                        "latency_ms": round(result.metrics.latency_ms, 2),
                        "throughput_rps": round(result.metrics.throughput_rps, 2),
                        "error_rate_percent": round(result.metrics.error_rate, 2)
                    },
                    "details": result.details,
                    "timestamp": result.timestamp
                }
                for result in self.results
            ],
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        self._save_report(report)
        
        logger.info(f"Validation completed: {passed_tests}/{total_tests} tests passed")
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Analyze latency
        latency_results = [r for r in self.results if "Latency" in r.test_name]
        if latency_results and not latency_results[0].passed:
            recommendations.append("Consider optimizing model inference or reducing data processing complexity")
            
        # Analyze load capacity
        load_results = [r for r in self.results if "Load" in r.test_name]
        if load_results and not load_results[0].passed:
            recommendations.append("Consider horizontal scaling or load balancing for better concurrency")
            
        # Analyze privacy guarantees
        privacy_results = [r for r in self.results if "Privacy" in r.test_name]
        if privacy_results and not privacy_results[0].passed:
            recommendations.append("Review privacy transformation parameters and ensure adequate privacy budget")
            
        if not recommendations:
            recommendations.append("All performance targets met - system is performing optimally")
            
        return recommendations
        
    def _save_report(self, report: Dict):
        """Save validation report to file"""
        report_dir = Path("validation_reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"performance_validation_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Validation report saved: {report_file}")

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QEP-VLA Performance Validator")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL for QEP-VLA API")
    parser.add_argument("--output", help="Output file for validation report")
    
    args = parser.parse_args()
    
    validator = PerformanceValidator(base_url=args.url)
    
    try:
        report = validator.run_full_validation()
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE VALIDATION SUMMARY")
        print("="*60)
        print(f"Overall Status: {report['validation_summary']['overall_status']}")
        print(f"Tests Passed: {report['validation_summary']['passed_tests']}/{report['validation_summary']['total_tests']}")
        print(f"Success Rate: {report['validation_summary']['success_rate']:.1f}%")
        print(f"Average Latency: {report['performance_metrics']['average_latency_ms']}ms")
        print(f"Average Throughput: {report['performance_metrics']['average_throughput_rps']} RPS")
        print(f"Average Error Rate: {report['performance_metrics']['average_error_rate_percent']:.1f}%")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
                
        print("="*60)
        
        # Exit with appropriate code
        if report['validation_summary']['overall_status'] == 'PASSED':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()
