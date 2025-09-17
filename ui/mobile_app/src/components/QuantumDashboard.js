import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Animated,
  Dimensions
} from 'react-native';
import { LineChart, BarChart } from 'react-native-chart-kit';
import Icon from 'react-native-vector-icons/MaterialIcons';

const { width } = Dimensions.get('window');

const QuantumDashboard = () => {
  const [navigationData, setNavigationData] = useState({
    accuracy: 97.3,
    latency: 47,
    privacyLevel: 'high',
    quantumEnhanced: true,
    vehicleStatus: 'active'
  });

  const [animatedValue] = useState(new Animated.Value(0));

  useEffect(() => {
    // Pulse animation for quantum indicators
    Animated.loop(
      Animated.sequence([
        Animated.timing(animatedValue, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(animatedValue, {
          toValue: 0,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);

  const pulseStyle = {
    opacity: animatedValue.interpolate({
      inputRange: [0, 1],
      outputRange: [0.5, 1],
    }),
    transform: [
      {
        scale: animatedValue.interpolate({
          inputRange: [0, 1],
          outputRange: [1, 1.05],
        }),
      },
    ],
  };

  const accuracyData = {
    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
    datasets: [
      {
        data: [96.8, 97.1, 97.5, 97.3, 97.2, 97.4],
        color: (opacity = 1) => `rgba(46, 204, 113, ${opacity})`,
        strokeWidth: 3,
      },
    ],
  };

  const latencyData = {
    labels: ['Nav', 'Proc', 'Enc', 'Dec', 'Resp'],
    datasets: [
      {
        data: [12, 15, 8, 7, 5],
      },
    ],
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Animated.View style={[styles.quantumIcon, pulseStyle]}>
          <Icon name="scatter-plot" size={30} color="#fff" />
        </Animated.View>
        <Text style={styles.headerTitle}>QEP-VLA Mobile</Text>
        <Text style={styles.headerSubtitle}>Quantum Navigation Control</Text>
      </View>

      {/* Key Metrics */}
      <View style={styles.metricsContainer}>
        <View style={styles.metricCard}>
          <Icon name="gps-fixed" size={24} color="#2ecc71" />
          <Text style={styles.metricValue}>{navigationData.accuracy}%</Text>
          <Text style={styles.metricLabel}>Accuracy</Text>
        </View>

        <View style={styles.metricCard}>
          <Icon name="speed" size={24} color="#3498db" />
          <Text style={styles.metricValue}>{navigationData.latency}ms</Text>
          <Text style={styles.metricLabel}>Latency</Text>
        </View>

        <View style={styles.metricCard}>
          <Icon name="security" size={24} color="#9b59b6" />
          <Text style={styles.metricValue}>ε=0.1</Text>
          <Text style={styles.metricLabel}>Privacy</Text>
        </View>
      </View>

      {/* Navigation Status */}
      <View style={styles.statusContainer}>
        <Text style={styles.sectionTitle}>Navigation Status</Text>
        <View style={styles.statusCard}>
          <View style={styles.statusIndicator}>
            <View 
              style={[
                styles.statusDot, 
                { backgroundColor: navigationData.vehicleStatus === 'active' ? '#2ecc71' : '#e74c3c' }
              ]} 
            />
            <Text style={styles.statusText}>
              {navigationData.vehicleStatus === 'active' ? 'Active Navigation' : 'Standby'}
            </Text>
          </View>
          
          <TouchableOpacity style={styles.emergencyButton}>
            <Icon name="warning" size={20} color="#fff" />
            <Text style={styles.emergencyText}>Emergency Stop</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Accuracy Chart */}
      <View style={styles.chartContainer}>
        <Text style={styles.sectionTitle}>Navigation Accuracy</Text>
        <LineChart
          data={accuracyData}
          width={width - 40}
          height={200}
          chartConfig={{
            backgroundColor: '#1e3c72',
            backgroundGradientFrom: '#1e3c72',
            backgroundGradientTo: '#2a5298',
            decimalPlaces: 1,
            color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
            labelColor: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
            style: {
              borderRadius: 16
            },
            propsForDots: {
              r: "4",
              strokeWidth: "2",
              stroke: "#2ecc71"
            }
          }}
          bezier
          style={styles.chart}
        />
      </View>

      {/* Latency Breakdown */}
      <View style={styles.chartContainer}>
        <Text style={styles.sectionTitle}>Processing Latency Breakdown</Text>
        <BarChart
          data={latencyData}
          width={width - 40}
          height={200}
          chartConfig={{
            backgroundColor: '#667eea',
            backgroundGradientFrom: '#667eea',
            backgroundGradientTo: '#764ba2',
            decimalPlaces: 0,
            color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
            labelColor: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
          }}
          style={styles.chart}
        />
      </View>

      {/* Privacy Controls */}
      <View style={styles.controlsContainer}>
        <Text style={styles.sectionTitle}>Privacy Controls</Text>
        
        <View style={styles.privacyControl}>
          <Text style={styles.controlLabel}>Privacy Level</Text>
          <View style={styles.privacyButtons}>
            {['high', 'medium', 'low'].map((level) => (
              <TouchableOpacity
                key={level}
                style={[
                  styles.privacyButton,
                  navigationData.privacyLevel === level && styles.privacyButtonActive
                ]}
                onPress={() => setNavigationData({...navigationData, privacyLevel: level})}
              >
                <Text style={[
                  styles.privacyButtonText,
                  navigationData.privacyLevel === level && styles.privacyButtonTextActive
                ]}>
                  {level.charAt(0).toUpperCase() + level.slice(1)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={styles.quantumControl}>
          <Text style={styles.controlLabel}>Quantum Enhancement</Text>
          <TouchableOpacity
            style={[
              styles.quantumToggle,
              navigationData.quantumEnhanced && styles.quantumToggleActive
            ]}
            onPress={() => setNavigationData({
              ...navigationData, 
              quantumEnhanced: !navigationData.quantumEnhanced
            })}
          >
            <Animated.View style={[styles.quantumToggleInner, pulseStyle]}>
              <Icon 
                name={navigationData.quantumEnhanced ? "radio-button-checked" : "radio-button-unchecked"} 
                size={20} 
                color={navigationData.quantumEnhanced ? "#2ecc71" : "#95a5a6"} 
              />
              <Text style={[
                styles.quantumToggleText,
                navigationData.quantumEnhanced && styles.quantumToggleTextActive
              ]}>
                {navigationData.quantumEnhanced ? 'Enabled' : 'Disabled'}
              </Text>
            </Animated.View>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f1419',
  },
  header: {
    backgroundColor: '#1e3c72',
    padding: 20,
    alignItems: 'center',
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
  },
  quantumIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 5,
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  metricsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 20,
  },
  metricCard: {
    backgroundColor: '#1a202c',
    padding: 15,
    borderRadius: 15,
    alignItems: 'center',
    minWidth: 100,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  metricValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 8,
  },
  metricLabel: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
    marginTop: 4,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 15,
  },
  statusContainer: {
    padding: 20,
  },
  statusCard: {
    backgroundColor: '#1a202c',
    padding: 15,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 10,
  },
  statusText: {
    fontSize: 16,
    color: '#fff',
  },
  emergencyButton: {
    backgroundColor: '#e74c3c',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 10,
  },
  emergencyText: {
    color: '#fff',
    fontWeight: 'bold',
    marginLeft: 8,
  },
  chartContainer: {
    padding: 20,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  controlsContainer: {
    padding: 20,
  },
  privacyControl: {
    marginBottom: 20,
  },
  controlLabel: {
    fontSize: 16,
    color: '#fff',
    marginBottom: 10,
  },
  privacyButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  privacyButton: {
    flex: 1,
    backgroundColor: '#1a202c',
    padding: 12,
    borderRadius: 10,
    marginHorizontal: 5,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  privacyButtonActive: {
    backgroundColor: '#2ecc71',
    borderColor: '#27ae60',
  },
  privacyButtonText: {
    color: 'rgba(255, 255, 255, 0.7)',
    fontSize: 14,
  },
  privacyButtonTextActive: {
    color: '#fff',
    fontWeight: 'bold',
  },
  quantumControl: {
    marginBottom: 20,
  },
  quantumToggle: {
    backgroundColor: '#1a202c',
    padding: 15,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  quantumToggleActive: {
    borderColor: '#2ecc71',
  },
  quantumToggleInner: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  quantumToggleText: {
    color: 'rgba(255, 255, 255, 0.7)',
    marginLeft: 10,
    fontSize: 16,
  },
  quantumToggleTextActive: {
    color: '#2ecc71',
    fontWeight: 'bold',
  },
});

export default QuantumDashboard;
