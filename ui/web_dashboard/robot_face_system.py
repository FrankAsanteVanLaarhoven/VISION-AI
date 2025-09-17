# Robot Face Recognition & Conversation System
# Complete implementation with OpenCV and face_recognition library

import cv2
import face_recognition
import sqlite3
import pickle
import numpy as np
import json
from datetime import datetime
import os
import time
import threading
import queue
import random
from typing import List, Tuple, Optional, Dict

class RobotFaceRecognitionSystem:
    """
    Advanced Robot Face Recognition System with Conversation Initiation
    
    Features:
    - Real-time face detection and recognition
    - Unknown face detection with name collection
    - Database storage with SQLite
    - Adaptive learning capabilities
    - Conversation initiation based on recognition
    - Encounter tracking and statistics
    """
    
    def __init__(self, db_path: str = "robot_faces.db", 
                 recognition_threshold: float = 0.6,
                 min_face_size: int = 40):
        """
        Initialize the robot face recognition system
        
        Args:
            db_path: Path to SQLite database
            recognition_threshold: Face recognition confidence threshold
            min_face_size: Minimum face size for detection
        """
        self.db_path = db_path
        self.recognition_threshold = recognition_threshold
        self.min_face_size = min_face_size
        
        # Face data storage
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        
        # OpenCV cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Conversation system
        self.conversation_templates = {
            'first_meeting': [
                "Hello! I don't think we've met before. What's your name?",
                "Hi there! I'd love to know what to call you.",
                "Nice to meet you! Could you tell me your name?",
                "Hello! I don't recognize you. What should I call you?"
            ],
            'returning_user': [
                "Hello {name}! Great to see you again! How are you doing today?",
                "Hi {name}! Welcome back! How has your day been?",
                "Hey {name}! Nice to see you. What brings you here today?",
                "Good to see you again, {name}! How are things?"
            ],
            'regular_user': [
                "Hello {name}! Always a pleasure to see you!",
                "Hi {name}! You're becoming a regular here!",
                "Welcome back, {name}! Hope you're having a great day!"
            ]
        }
        
        # System parameters
        self.conversation_cooldown = 30  # seconds between greetings
        self.last_conversation_time = {}
        
        # Initialize system
        self._init_database()
        self._load_face_data()
        
        print(f"🤖 Robot Face Recognition System initialized")
        print(f"📊 Loaded {len(self.known_names)} known faces")
    
    def _init_database(self):
        """Initialize SQLite database with all necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Faces table - stores face encodings and metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                encoding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                encounter_count INTEGER DEFAULT 1,
                total_conversation_time INTEGER DEFAULT 0,
                personality_notes TEXT,
                preferences TEXT
            )
        ''')
        
        # Interactions table - conversation and interaction history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                interaction_type TEXT NOT NULL,
                conversation_snippet TEXT,
                confidence_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_seconds INTEGER DEFAULT 0,
                FOREIGN KEY (face_id) REFERENCES faces (id)
            )
        ''')
        
        # Learning data table - for adaptive behavior
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                feature_type TEXT,
                feature_data TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (face_id) REFERENCES faces (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("🗄️ Database initialized successfully")
    
    def _load_face_data(self):
        """Load all known faces from database into memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, encoding FROM faces")
        results = cursor.fetchall()
        
        self.known_ids = []
        self.known_names = []
        self.known_encodings = []
        
        for face_id, name, encoding_blob in results:
            try:
                encoding = pickle.loads(encoding_blob)
                self.known_ids.append(face_id)
                self.known_names.append(name)
                self.known_encodings.append(encoding)
            except Exception as e:
                print(f"⚠️ Error loading encoding for {name}: {e}")
        
        conn.close()
        print(f"📥 Loaded {len(self.known_names)} faces from database")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the given frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face locations as (top, right, bottom, left)
        """
        # Convert to RGB for face_recognition (it expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use face_recognition library for better accuracy
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        return face_locations
    
    def recognize_faces(self, frame: np.ndarray, face_locations: List) -> List[Dict]:
        """
        Recognize faces in the frame
        
        Args:
            frame: Input image frame
            face_locations: List of detected face locations
            
        Returns:
            List of recognition results
        """
        if not face_locations or len(self.known_encodings) == 0:
            return []
        
        # Get face encodings for detected faces
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognition_results = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_encodings, face_encoding, tolerance=self.recognition_threshold
            )
            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            
            result = {
                'location': face_location,
                'encoding': face_encoding,
                'name': 'Unknown',
                'confidence': 0.0,
                'face_id': None,
                'is_known': False
            }
            
            if True in matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    result['name'] = self.known_names[best_match_index]
                    result['confidence'] = 1 - face_distances[best_match_index]
                    result['face_id'] = self.known_ids[best_match_index]
                    result['is_known'] = True
            
            recognition_results.append(result)
        
        return recognition_results
    
    def save_new_face(self, name: str, face_encoding: np.ndarray, notes: str = "") -> Optional[int]:
        """
        Save a new face to the database
        
        Args:
            name: Person's name
            face_encoding: Face encoding array
            notes: Additional notes
            
        Returns:
            Face ID if successful, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Convert encoding to blob
            encoding_blob = pickle.dumps(face_encoding)
            
            cursor.execute('''
                INSERT INTO faces (name, encoding, personality_notes) 
                VALUES (?, ?, ?)
            ''', (name, encoding_blob, notes))
            
            face_id = cursor.lastrowid
            
            # Log the first meeting
            cursor.execute('''
                INSERT INTO interactions (face_id, interaction_type, conversation_snippet)
                VALUES (?, ?, ?)
            ''', (face_id, 'first_meeting', f'Initial meeting with {name}'))
            
            conn.commit()
            
            # Update memory
            self.known_ids.append(face_id)
            self.known_names.append(name)
            self.known_encodings.append(face_encoding)
            
            print(f"💾 Saved new face: {name} (ID: {face_id})")
            return face_id
            
        except sqlite3.IntegrityError:
            print(f"⚠️ Person named {name} already exists")
            return None
        except Exception as e:
            print(f"❌ Error saving face: {e}")
            return None
        finally:
            conn.close()
    
    def update_encounter(self, face_id: int, interaction_type: str = "recognition", 
                        conversation: str = "", confidence: float = 0.0):
        """Update encounter information for a known face"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update face record
        cursor.execute('''
            UPDATE faces 
            SET last_seen = CURRENT_TIMESTAMP, encounter_count = encounter_count + 1
            WHERE id = ?
        ''', (face_id,))
        
        # Log interaction
        cursor.execute('''
            INSERT INTO interactions (face_id, interaction_type, conversation_snippet, confidence_score)
            VALUES (?, ?, ?, ?)
        ''', (face_id, interaction_type, conversation, confidence))
        
        conn.commit()
        conn.close()
    
    def generate_conversation(self, recognition_result: Dict) -> str:
        """
        Generate appropriate conversation based on recognition result
        
        Args:
            recognition_result: Dictionary containing recognition information
            
        Returns:
            Generated conversation string
        """
        if not recognition_result['is_known']:
            return random.choice(self.conversation_templates['first_meeting'])
        
        name = recognition_result['name']
        face_id = recognition_result['face_id']
        
        # Get encounter count to determine conversation type
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT encounter_count FROM faces WHERE id = ?", (face_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            encounter_count = result[0]
            
            if encounter_count <= 3:
                template = random.choice(self.conversation_templates['returning_user'])
            else:
                template = random.choice(self.conversation_templates['regular_user'])
            
            return template.format(name=name)
        
        return f"Hello {name}! Good to see you!"
    
    def handle_unknown_face(self, frame: np.ndarray, recognition_result: Dict) -> Optional[str]:
        """
        Handle unknown face detection - ask for name and save
        
        Args:
            frame: Current video frame
            recognition_result: Recognition result dictionary
            
        Returns:
            Name if successful, None otherwise
        """
        print(f"\n❓ Unknown face detected!")
        print(f"🤖 ROBOT: {random.choice(self.conversation_templates['first_meeting'])}")
        
        # In a real robot, this would use speech recognition
        # For now, we'll use console input
        try:
            name = input("👤 Please enter your name: ").strip()
            
            if name and len(name) > 0:
                face_id = self.save_new_face(name, recognition_result['encoding'], 
                                           "Added through unknown face detection")
                
                if face_id:
                    welcome_message = f"Nice to meet you, {name}! I'll remember you next time."
                    print(f"🤖 ROBOT: {welcome_message}")
                    
                    # Update the result
                    recognition_result['name'] = name
                    recognition_result['face_id'] = face_id
                    recognition_result['is_known'] = True
                    
                    return name
            
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error handling unknown face: {e}")
        
        return None
    
    def should_initiate_conversation(self, name: str) -> bool:
        """
        Check if enough time has passed to initiate conversation again
        
        Args:
            name: Person's name
            
        Returns:
            True if conversation should be initiated
        """
        current_time = time.time()
        
        if name in self.last_conversation_time:
            time_since_last = current_time - self.last_conversation_time[name]
            return time_since_last >= self.conversation_cooldown
        
        return True
    
    def draw_recognition_results(self, frame: np.ndarray, recognition_results: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and names on the frame
        
        Args:
            frame: Input frame
            recognition_results: List of recognition results
            
        Returns:
            Frame with annotations
        """
        for result in recognition_results:
            top, right, bottom, left = result['location']
            name = result['name']
            confidence = result['confidence']
            is_known = result['is_known']
            
            # Choose color based on recognition status
            color = (0, 255, 0) if is_known else (0, 0, 255)  # Green for known, red for unknown
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Create label
            if is_known:
                label = f"{name} ({confidence:.2f})"
            else:
                label = "Unknown"
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run_face_recognition_system(self, camera_index: int = 0, window_name: str = "Robot Face Recognition"):
        """
        Main loop for the face recognition system
        
        Args:
            camera_index: Camera device index
            window_name: OpenCV window name
        """
        print("🚀 Starting Robot Face Recognition System...")
        print("📷 Initializing camera...")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized successfully")
        print("\n🎮 Controls:")
        print("   'q' - Quit system")
        print("   'r' - Reload face database")
        print("   's' - Show statistics")
        print("   'c' - Clear conversation cooldowns")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading from camera")
                    break
                
                frame_count += 1
                
                # Process every 3rd frame for performance
                if frame_count % 3 == 0:
                    # Detect faces
                    face_locations = self.detect_faces(frame)
                    
                    if face_locations:
                        # Recognize faces
                        recognition_results = self.recognize_faces(frame, face_locations)
                        
                        # Process each recognized face
                        for result in recognition_results:
                            name = result['name']
                            
                            if result['is_known']:
                                # Known face - check if we should greet
                                if self.should_initiate_conversation(name):
                                    conversation = self.generate_conversation(result)
                                    print(f"\n🤖 ROBOT: {conversation}")
                                    
                                    # Update encounter
                                    self.update_encounter(
                                        result['face_id'], 
                                        "recognition", 
                                        conversation, 
                                        result['confidence']
                                    )
                                    
                                    # Update conversation time
                                    self.last_conversation_time[name] = time.time()
                            
                            else:
                                # Unknown face - handle if not in cooldown
                                if self.should_initiate_conversation("Unknown"):
                                    new_name = self.handle_unknown_face(frame, result)
                                    if new_name:
                                        self.last_conversation_time[new_name] = time.time()
                                    else:
                                        self.last_conversation_time["Unknown"] = time.time()
                        
                        # Draw results on frame
                        frame = self.draw_recognition_results(frame, recognition_results)
                    
                # Add system info to frame
                info_text = f"Known faces: {len(self.known_names)} | Frame: {frame_count}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Shutting down system...")
                    break
                elif key == ord('r'):
                    print("\n🔄 Reloading face database...")
                    self._load_face_data()
                elif key == ord('s'):
                    self.show_statistics()
                elif key == ord('c'):
                    print("\n🧹 Clearing conversation cooldowns...")
                    self.last_conversation_time.clear()
        
        except KeyboardInterrupt:
            print("\n⏹️ System interrupted by user")
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("✅ System shutdown complete")
    
    def show_statistics(self):
        """Display system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get face statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_faces,
                AVG(encounter_count) as avg_encounters,
                MAX(encounter_count) as max_encounters
            FROM faces
        ''')
        face_stats = cursor.fetchone()
        
        # Get interaction count
        cursor.execute('SELECT COUNT(*) FROM interactions')
        total_interactions = cursor.fetchone()[0]
        
        # Get top recognized faces
        cursor.execute('''
            SELECT name, encounter_count, last_seen 
            FROM faces 
            ORDER BY encounter_count DESC 
            LIMIT 5
        ''')
        top_faces = cursor.fetchall()
        
        conn.close()
        
        print("\n📊 System Statistics")
        print("=" * 50)
        print(f"👥 Total Known Faces: {face_stats[0]}")
        print(f"💬 Total Interactions: {total_interactions}")
        if face_stats[1]:
            print(f"📈 Average Encounters: {face_stats[1]:.1f}")
            print(f"🏆 Max Encounters: {face_stats[2]}")
        
        if top_faces:
            print("\n🌟 Most Recognized People:")
            for name, count, last_seen in top_faces:
                print(f"   👤 {name}: {count} encounters (last: {last_seen})")
        print("=" * 50)

def main():
    """Main function to run the robot face recognition system"""
    print("🤖 Robot Face Recognition & Conversation System")
    print("=" * 60)
    print("⚙️ Initializing system...")
    
    try:
        # Create the face recognition system
        robot = RobotFaceRecognitionSystem(
            db_path="robot_faces.db",
            recognition_threshold=0.6,
            min_face_size=40
        )
        
        # Run the main system
        robot.run_face_recognition_system()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("🔧 Make sure you have installed:")
        print("   pip install opencv-python face-recognition numpy")
    except Exception as e:
        print(f"❌ System Error: {e}")
        print("🔧 Check your camera and system configuration")

if __name__ == "__main__":
    main()
