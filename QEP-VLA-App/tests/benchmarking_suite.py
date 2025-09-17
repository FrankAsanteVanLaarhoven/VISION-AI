"""
Benchmarking Suite for QEP-VLA Application
Comprehensive performance and scalability testing
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import time
import json
import statistics
from datetime import datetime
import psutil
import gc

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantum_privacy_transform import QuantumPrivacyTransform, QuantumTransformConfig, QuantumTransformType
from core.navigation_engine import NavigationEngine, NavigationConfig, NavigationMode
from core.scenario_generator import ScenarioGenerator, ScenarioConfig, ScenarioType
from core.secure_aggregation import SecureAggregation

class BenchmarkSuite:
    """Comprehensive benchmarking suite for QEP-VLA system"""
    
    def __init__(self):
        self.results = {}
        self.benchmark_start_time = None
        
    def run_all_benchmarks(self):
        """Run all benchmark tests"""
        self.benchmark_start_time = datetime.now()
        print(f"🚀 Starting QEP-VLA Benchmark Suite at {self.benchmark_start_time}")
        
        # Run component benchmarks
        self._benchmark_quantum_privacy_transform()
        self._benchmark_navigation_engine()
        self._benchmark_scenario_generator()
        self._benchmark_secure_aggregation()
        
        # Run system benchmarks
        self._benchmark_system_performance()
        self._benchmark_memory_usage()
        self._benchmark_scalability()
        
        # Generate report
        self._generate_benchmark_report()
        
        print(f"✅ Benchmark suite completed in {datetime.now() - self.benchmark_start_time}")
    
    def _benchmark_quantum_privacy_transform(self):
        """Benchmark quantum privacy transformation performance"""
        print("\n🔐 Benchmarking Quantum Privacy Transform...")
        
        config = QuantumTransformConfig()
        transform = QuantumPrivacyTransform(config)
        
        # Test data sizes
        data_sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
        transform_types = list(QuantumTransformType)
        
        results = {}
        
        for size in data_sizes:
            print(f"  Testing data size: {size[0]}x{size[1]}")
            size_results = {}
            
            # Generate test data
            test_data = np.random.rand(*size)
            
            for transform_type in transform_types:
                print(f"    Testing {transform_type.value}...")
                
                # Warm up
                for _ in range(3):
                    transform.apply_transform(test_data, transform_type)
                
                # Benchmark
                times = []
                memory_usage = []
                
                for _ in range(10):
                    # Measure memory before
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Measure time
                    start_time = time.time()
                    result = transform.apply_transform(test_data, transform_type)
                    end_time = time.time()
                    
                    # Measure memory after
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    
                    times.append(end_time - start_time)
                    memory_usage.append(memory_after - memory_before)
                    
                    # Verify result
                    self._verify_transformation_result(result, test_data)
                
                # Calculate statistics
                size_results[transform_type.value] = {
                    'mean_time': statistics.mean(times),
                    'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                    'min_time': min(times),
                    'max_time': max(times),
                    'mean_memory': statistics.mean(memory_usage),
                    'std_memory': statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
                    'throughput': (size[0] * size[1]) / statistics.mean(times)  # pixels per second
                }
            
            results[f"{size[0]}x{size[1]}"] = size_results
        
        self.results['quantum_privacy_transform'] = results
        print("  ✅ Quantum Privacy Transform benchmarking completed")
    
    def _benchmark_navigation_engine(self):
        """Benchmark navigation engine performance"""
        print("\n🧭 Benchmarking Navigation Engine...")
        
        config = NavigationConfig()
        nav_engine = NavigationEngine(config)
        
        # Test scenarios
        test_scenarios = [
            {'start': [0, 0, 0], 'target': [10, 0, 0], 'obstacles': 0, 'privacy_zones': 0},
            {'start': [0, 0, 0], 'target': [20, 0, 0], 'obstacles': 5, 'privacy_zones': 3},
            {'start': [0, 0, 0], 'target': [50, 0, 0], 'obstacles': 20, 'privacy_zones': 10},
            {'start': [0, 0, 0], 'target': [100, 0, 0], 'obstacles': 50, 'privacy_zones': 25}
        ]
        
        results = {}
        
        for i, scenario in enumerate(test_scenarios):
            print(f"  Testing scenario {i+1}: {scenario['obstacles']} obstacles, {scenario['privacy_zones']} privacy zones")
            
            # Set up scenario
            nav_engine.set_current_position(np.array(scenario['start']), 0.0)
            
            # Add obstacles
            for j in range(scenario['obstacles']):
                pos = np.array([5 + j * 2, np.random.uniform(-5, 5), 0])
                nav_engine.add_obstacle(pos, 1.0, 0.5)
            
            # Add privacy zones
            for j in range(scenario['privacy_zones']):
                pos = np.array([10 + j * 3, np.random.uniform(-8, 8), 0])
                nav_engine.add_privacy_zone(pos, 3.0, 0.8)
            
            # Benchmark path planning
            modes = [NavigationMode.PRIVACY_AWARE, NavigationMode.EXPLORATION, NavigationMode.OBSTACLE_AVOIDANCE]
            mode_results = {}
            
            for mode in modes:
                print(f"    Testing {mode.value} mode...")
                
                times = []
                path_lengths = []
                
                for _ in range(10):
                    start_time = time.time()
                    path = nav_engine.plan_path(np.array(scenario['target']), mode)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    path_lengths.append(len(path))
                    
                    # Verify path
                    self._verify_navigation_path(path, scenario['start'], scenario['target'])
                
                # Calculate statistics
                mode_results[mode.value] = {
                    'mean_time': statistics.mean(times),
                    'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                    'mean_path_length': statistics.mean(path_lengths),
                    'std_path_length': statistics.stdev(path_lengths) if len(path_lengths) > 1 else 0,
                    'throughput': 1 / statistics.mean(times)  # paths per second
                }
            
            results[f"scenario_{i+1}"] = mode_results
            
            # Clear scenario
            nav_engine.obstacles.clear()
            nav_engine.privacy_zones.clear()
        
        self.results['navigation_engine'] = results
        print("  ✅ Navigation Engine benchmarking completed")
    
    def _benchmark_scenario_generator(self):
        """Benchmark scenario generator performance"""
        print("\n🎭 Benchmarking Scenario Generator...")
        
        config = ScenarioConfig()
        scenario_gen = ScenarioGenerator(config)
        
        # Test scenario types
        scenario_types = list(ScenarioType)
        results = {}
        
        for scenario_type in scenario_types:
            print(f"  Testing {scenario_type.value}...")
            
            times = []
            scenario_sizes = []
            
            for _ in range(10):
                start_time = time.time()
                scenario = scenario_gen.generate_scenario(scenario_type)
                end_time = time.time()
                
                times.append(end_time - start_time)
                
                # Measure scenario complexity
                if 'privacy_zones' in scenario:
                    scenario_sizes.append(len(scenario['privacy_zones']))
                elif 'obstacles' in scenario:
                    scenario_sizes.append(len(scenario['obstacles']))
                elif 'agents' in scenario:
                    scenario_sizes.append(len(scenario['agents']))
                else:
                    scenario_sizes.append(0)
                
                # Verify scenario
                self._verify_generated_scenario(scenario, scenario_type)
            
            # Calculate statistics
            results[scenario_type.value] = {
                'mean_time': statistics.mean(times),
                'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                'mean_size': statistics.mean(scenario_sizes),
                'std_size': statistics.stdev(scenario_sizes) if len(scenario_sizes) > 1 else 0,
                'throughput': 1 / statistics.mean(times)  # scenarios per second
            }
        
        self.results['scenario_generator'] = results
        print("  ✅ Scenario Generator benchmarking completed")
    
    def _benchmark_secure_aggregation(self):
        """Benchmark secure aggregation performance"""
        print("\n🔒 Benchmarking Secure Aggregation...")
        
        config = {'security_level': 'high', 'differential_privacy': True, 'encryption_enabled': True}
        secure_agg = SecureAggregation(config)
        
        # Test client configurations
        client_configs = [5, 10, 20, 50]
        results = {}
        
        for num_clients in client_configs:
            print(f"  Testing {num_clients} clients...")
            
            # Add clients
            for i in range(num_clients):
                # Mock public key
                mock_key = b'mock_public_key_' + str(i).encode()
                secure_agg.add_client(f"client_{i}", mock_key)
            
            # Generate mock updates
            update_size = (100, 100)  # 100x100 tensor
            client_updates = {}
            
            for i in range(num_clients):
                client_updates[f"client_{i}"] = {
                    'layer1.weight': np.random.randn(*update_size),
                    'layer1.bias': np.random.randn(100),
                    'layer2.weight': np.random.randn(100, 10),
                    'layer2.bias': np.random.randn(10)
                }
            
            # Benchmark aggregation methods
            methods = ['weighted_average', 'median', 'trimmed_mean']
            method_results = {}
            
            for method in methods:
                print(f"    Testing {method} aggregation...")
                
                times = []
                
                for _ in range(5):
                    start_time = time.time()
                    aggregated = secure_agg.secure_aggregate(client_updates, method)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    
                    # Verify aggregation
                    self._verify_secure_aggregation(aggregated, client_updates)
                
                # Calculate statistics
                method_results[method] = {
                    'mean_time': statistics.mean(times),
                    'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                    'throughput': 1 / statistics.mean(times)  # aggregations per second
                }
            
            results[f"{num_clients}_clients"] = method_results
        
        self.results['secure_aggregation'] = results
        print("  ✅ Secure Aggregation benchmarking completed")
    
    def _benchmark_system_performance(self):
        """Benchmark overall system performance"""
        print("\n⚡ Benchmarking System Performance...")
        
        # CPU and memory monitoring
        process = psutil.Process()
        
        # Baseline measurements
        baseline_cpu = psutil.cpu_percent(interval=1)
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Stress test
        print("  Running stress test...")
        
        # Create multiple components
        components = []
        for i in range(10):
            config = QuantumTransformConfig()
            transform = QuantumPrivacyTransform(config)
            components.append(transform)
        
        # Run operations
        start_time = time.time()
        operations = 0
        
        for _ in range(100):
            for component in components:
                test_data = np.random.rand(50, 50)
                component.apply_transform(test_data, QuantumTransformType.QUANTUM_NOISE)
                operations += 1
        
        end_time = time.time()
        
        # Final measurements
        final_cpu = psutil.cpu_percent(interval=1)
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        total_time = end_time - start_time
        operations_per_second = operations / total_time
        
        self.results['system_performance'] = {
            'baseline_cpu': baseline_cpu,
            'baseline_memory': baseline_memory,
            'final_cpu': final_cpu,
            'final_memory': final_memory,
            'cpu_increase': final_cpu - baseline_cpu,
            'memory_increase': final_memory - baseline_memory,
            'operations_per_second': operations_per_second,
            'total_operations': operations,
            'total_time': total_time
        }
        
        print("  ✅ System Performance benchmarking completed")
    
    def _benchmark_memory_usage(self):
        """Benchmark memory usage patterns"""
        print("\n💾 Benchmarking Memory Usage...")
        
        # Test memory allocation patterns
        memory_tests = []
        
        for test_size in [100, 500, 1000, 2000]:
            print(f"  Testing memory allocation for {test_size}x{test_size} data...")
            
            # Measure memory before
            gc.collect()
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Allocate and process data
            data = np.random.rand(test_size, test_size)
            
            # Apply transformations
            config = QuantumTransformConfig()
            transform = QuantumPrivacyTransform(config)
            
            for _ in range(5):
                result = transform.apply_transform(data, QuantumTransformType.QUANTUM_NOISE)
                del result  # Explicitly delete
            
            # Measure memory after
            gc.collect()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_tests.append({
                'data_size': test_size,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_increase': memory_after - memory_before,
                'memory_per_pixel': (memory_after - memory_before) / (test_size * test_size)
            })
        
        self.results['memory_usage'] = memory_tests
        print("  ✅ Memory Usage benchmarking completed")
    
    def _benchmark_scalability(self):
        """Benchmark system scalability"""
        print("\n📈 Benchmarking System Scalability...")
        
        # Test scalability with increasing data sizes
        scalability_tests = []
        
        for data_size in [100, 200, 400, 800, 1600]:
            print(f"  Testing scalability with {data_size}x{data_size} data...")
            
            # Generate test data
            test_data = np.random.rand(data_size, data_size)
            
            # Test different components
            component_times = {}
            
            # Quantum Privacy Transform
            config = QuantumTransformConfig()
            transform = QuantumPrivacyTransform(config)
            
            start_time = time.time()
            result = transform.apply_transform(test_data, QuantumTransformType.QUANTUM_NOISE)
            end_time = time.time()
            component_times['quantum_privacy_transform'] = end_time - start_time
            
            # Navigation Engine (simplified test)
            nav_config = NavigationConfig()
            nav_engine = NavigationEngine(nav_config)
            nav_engine.set_current_position(np.array([0, 0, 0]), 0.0)
            
            start_time = time.time()
            path = nav_engine.plan_path(np.array([data_size/10, 0, 0]))
            end_time = time.time()
            component_times['navigation_engine'] = end_time - start_time
            
            # Calculate scalability metrics
            scalability_tests.append({
                'data_size': data_size,
                'pixel_count': data_size * data_size,
                'component_times': component_times,
                'throughput_pixels_per_second': (data_size * data_size) / component_times['quantum_privacy_transform']
            })
        
        self.results['scalability'] = scalability_tests
        print("  ✅ System Scalability benchmarking completed")
    
    def _verify_transformation_result(self, result, original_data):
        """Verify transformation result validity"""
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        assert result.shape == original_data.shape, "Result should have same shape as input"
        assert not np.array_equal(result, original_data), "Result should be different from input"
    
    def _verify_navigation_path(self, path, start, target):
        """Verify navigation path validity"""
        assert isinstance(path, list), "Path should be a list"
        assert len(path) > 0, "Path should not be empty"
        assert np.array_equal(path[0], np.array(start)), "Path should start at start position"
        assert np.array_equal(path[-1], np.array(target)), "Path should end at target position"
    
    def _verify_generated_scenario(self, scenario, scenario_type):
        """Verify generated scenario validity"""
        assert isinstance(scenario, dict), "Scenario should be a dictionary"
        assert 'scenario_type' in scenario, "Scenario should have type field"
        assert scenario['scenario_type'] == scenario_type.value, "Scenario type should match"
    
    def _verify_secure_aggregation(self, aggregated, client_updates):
        """Verify secure aggregation result validity"""
        assert isinstance(aggregated, dict), "Aggregated result should be a dictionary"
        assert len(aggregated) > 0, "Aggregated result should not be empty"
    
    def _generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        print("\n📊 Generating Benchmark Report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_duration': str(datetime.now() - self.benchmark_start_time),
            'summary': self._generate_summary(),
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_filename = f"qep_vla_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"  📄 Report saved to: {report_filename}")
        
        # Print summary
        self._print_summary(report['summary'])
    
    def _generate_summary(self):
        """Generate benchmark summary"""
        summary = {}
        
        # Quantum Privacy Transform summary
        if 'quantum_privacy_transform' in self.results:
            qpt_results = self.results['quantum_privacy_transform']
            summary['quantum_privacy_transform'] = {
                'fastest_transform': self._find_fastest_transform(qpt_results),
                'most_memory_efficient': self._find_most_memory_efficient(qpt_results),
                'best_throughput': self._find_best_throughput(qpt_results)
            }
        
        # Navigation Engine summary
        if 'navigation_engine' in self.results:
            nav_results = self.results['navigation_engine']
            summary['navigation_engine'] = {
                'fastest_mode': self._find_fastest_navigation_mode(nav_results),
                'most_efficient_paths': self._find_most_efficient_paths(nav_results)
            }
        
        # Overall performance summary
        if 'system_performance' in self.results:
            perf = self.results['system_performance']
            summary['overall_performance'] = {
                'operations_per_second': perf['operations_per_second'],
                'memory_efficiency': perf['memory_increase'] / perf['total_operations'],
                'cpu_efficiency': perf['cpu_increase'] / perf['total_operations']
            }
        
        return summary
    
    def _find_fastest_transform(self, qpt_results):
        """Find fastest transformation type"""
        fastest = None
        best_time = float('inf')
        
        for size, transforms in qpt_results.items():
            for transform_type, metrics in transforms.items():
                if metrics['mean_time'] < best_time:
                    best_time = metrics['mean_time']
                    fastest = f"{transform_type} ({size})"
        
        return fastest
    
    def _find_most_memory_efficient(self, qpt_results):
        """Find most memory efficient transformation"""
        most_efficient = None
        best_memory = float('inf')
        
        for size, transforms in qpt_results.items():
            for transform_type, metrics in transforms.items():
                if metrics['mean_memory'] < best_memory:
                    best_memory = metrics['mean_memory']
                    most_efficient = f"{transform_type} ({size})"
        
        return most_efficient
    
    def _find_best_throughput(self, qpt_results):
        """Find transformation with best throughput"""
        best = None
        best_throughput = 0
        
        for size, transforms in qpt_results.items():
            for transform_type, metrics in transforms.items():
                if metrics['throughput'] > best_throughput:
                    best_throughput = metrics['throughput']
                    best = f"{transform_type} ({size})"
        
        return best
    
    def _find_fastest_navigation_mode(self, nav_results):
        """Find fastest navigation mode"""
        fastest = None
        best_time = float('inf')
        
        for scenario, modes in nav_results.items():
            for mode, metrics in modes.items():
                if metrics['mean_time'] < best_time:
                    best_time = metrics['mean_time']
                    fastest = f"{mode} ({scenario})"
        
        return fastest
    
    def _find_most_efficient_paths(self, nav_results):
        """Find navigation mode with most efficient paths"""
        most_efficient = None
        best_length = float('inf')
        
        for scenario, modes in nav_results.items():
            for mode, metrics in modes.items():
                if metrics['mean_path_length'] < best_length:
                    best_length = metrics['mean_path_length']
                    most_efficient = f"{mode} ({scenario})"
        
        return most_efficient
    
    def _generate_recommendations(self):
        """Generate performance recommendations"""
        recommendations = []
        
        # Analyze results and generate recommendations
        if 'quantum_privacy_transform' in self.results:
            qpt_results = self.results['quantum_privacy_transform']
            
            # Check for performance bottlenecks
            for size, transforms in qpt_results.items():
                for transform_type, metrics in transforms.items():
                    if metrics['mean_time'] > 1.0:
                        recommendations.append({
                            'component': 'quantum_privacy_transform',
                            'issue': f'Slow performance for {transform_type} with {size} data',
                            'recommendation': 'Consider optimizing algorithm or using hardware acceleration',
                            'severity': 'medium'
                        })
        
        if 'system_performance' in self.results:
            perf = self.results['system_performance']
            
            if perf['memory_increase'] > 100:
                recommendations.append({
                    'component': 'system',
                    'issue': 'High memory usage increase',
                    'recommendation': 'Implement memory pooling and better garbage collection',
                    'severity': 'high'
                })
        
        return recommendations
    
    def _print_summary(self, summary):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("🏆 BENCHMARK SUMMARY")
        print("="*60)
        
        if 'quantum_privacy_transform' in summary:
            qpt = summary['quantum_privacy_transform']
            print(f"🔐 Quantum Privacy Transform:")
            print(f"   Fastest: {qpt['fastest_transform']}")
            print(f"   Most Memory Efficient: {qpt['most_memory_efficient']}")
            print(f"   Best Throughput: {qpt['best_throughput']}")
        
        if 'navigation_engine' in summary:
            nav = summary['navigation_engine']
            print(f"🧭 Navigation Engine:")
            print(f"   Fastest Mode: {nav['fastest_mode']}")
            print(f"   Most Efficient Paths: {nav['most_efficient_paths']}")
        
        if 'overall_performance' in summary:
            perf = summary['overall_performance']
            print(f"⚡ Overall Performance:")
            print(f"   Operations/Second: {perf['operations_per_second']:.2f}")
            print(f"   Memory Efficiency: {perf['memory_efficiency']:.6f} MB/op")
            print(f"   CPU Efficiency: {perf['cpu_efficiency']:.6f} %/op")
        
        print("="*60)

def run_benchmarks():
    """Run the complete benchmark suite"""
    suite = BenchmarkSuite()
    suite.run_all_benchmarks()
    return suite.results

if __name__ == '__main__':
    # Run benchmarks
    results = run_benchmarks()
    print(f"\n🎯 Benchmark suite completed successfully!")
    print(f"📊 Results available in the results dictionary")
