#!/usr/bin/env python3
"""
Simple Example Script for Robot Face Recognition System
This demonstrates basic usage without needing a camera
"""

import sqlite3
import numpy as np
import pickle
from datetime import datetime
import time

class SimpleFaceDemo:
    """Simplified demo version of the face recognition system"""
    
    def __init__(self, db_path="demo_faces.db"):
        self.db_path = db_path
        self.init_database()
        self.known_encodings = []
        self.known_names = []
        
    def init_database(self):
        """Initialize demo database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                encoding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                encounter_count INTEGER DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                greeting TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (face_id) REFERENCES faces (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Demo database initialized")
    
    def add_demo_person(self, name):
        """Add a demo person with simulated face encoding"""
        # Create a random 128-dimensional face encoding (simulating real face data)
        face_encoding = np.random.rand(128).astype(np.float64)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            encoding_blob = pickle.dumps(face_encoding)
            cursor.execute('''
                INSERT INTO faces (name, encoding) VALUES (?, ?)
            ''', (name, encoding_blob))
            
            face_id = cursor.lastrowid
            
            # Log first meeting
            cursor.execute('''
                INSERT INTO interactions (face_id, greeting)
                VALUES (?, ?)
            ''', (face_id, f"Nice to meet you, {name}!"))
            
            conn.commit()
            print(f"👤 Added {name} to face database")
            
            # Update memory
            self.known_names.append(name)
            self.known_encodings.append(face_encoding)
            
            return face_id
            
        except sqlite3.IntegrityError:
            print(f"⚠️ {name} already exists")
            return None
        finally:
            conn.close()
    
    def simulate_face_recognition(self, person_name):
        """Simulate recognizing a face"""
        if person_name in self.known_names:
            # Simulate recognition
            print(f"✅ Recognized: {person_name}")
            
            # Generate greeting based on encounter count
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current encounter count
            cursor.execute("SELECT encounter_count FROM faces WHERE name = ?", (person_name,))
            result = cursor.fetchone()
            
            if result:
                encounter_count = result[0]
                
                # Update encounter count
                cursor.execute('''
                    UPDATE faces SET encounter_count = encounter_count + 1
                    WHERE name = ?
                ''', (person_name,))
                
                # Generate appropriate greeting
                if encounter_count == 1:
                    greeting = f"Hello {person_name}! Great to see you again!"
                elif encounter_count < 5:
                    greeting = f"Hi {person_name}! Welcome back! How are you today?"
                else:
                    greeting = f"Hello {person_name}! Always a pleasure to see you!"
                
                print(f"🤖 ROBOT: {greeting}")
                
                # Log interaction
                cursor.execute('''
                    INSERT INTO interactions (face_id, greeting)
                    VALUES ((SELECT id FROM faces WHERE name = ?), ?)
                ''', (person_name, greeting))
                
                conn.commit()
            
            conn.close()
            
        else:
            # Unknown person
            print(f"❓ Unknown person detected!")
            print(f"🤖 ROBOT: Hello! I don't think we've met. What's your name?")
            print(f"👤 USER: Hi, I'm {person_name}")
            
            # Add them to database
            face_id = self.add_demo_person(person_name)
            if face_id:
                print(f"🤖 ROBOT: Nice to meet you, {person_name}! I'll remember you next time.")
    
    def show_statistics(self):
        """Show system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get face count and interactions
        cursor.execute("SELECT COUNT(*) FROM faces")
        face_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM interactions")
        interaction_count = cursor.fetchone()[0]
        
        # Get top people
        cursor.execute('''
            SELECT name, encounter_count 
            FROM faces 
            ORDER BY encounter_count DESC 
            LIMIT 5
        ''')
        top_people = cursor.fetchall()
        
        print("\n📊 Demo Statistics")
        print("=" * 40)
        print(f"👥 Known People: {face_count}")
        print(f"💬 Total Interactions: {interaction_count}")
        
        if top_people:
            print("\n🌟 Most Encountered:")
            for name, count in top_people:
                print(f"   👤 {name}: {count} times")
        
        conn.close()
    
    def run_demo(self):
        """Run the demo simulation"""
        print("🤖 Face Recognition Demo Starting...")
        print("=" * 50)
        
        # Simulate various scenarios
        scenarios = [
            "Alice",      # New person
            "Bob",        # Another new person  
            "Alice",      # Alice returns
            "Charlie",    # Third new person
            "Bob",        # Bob returns
            "Alice",      # Alice becomes regular
            "Diana",      # Fourth new person
            "Alice",      # Alice is now a regular
        ]
        
        for i, person in enumerate(scenarios, 1):
            print(f"\n📹 Scenario {i}: Camera detects someone...")
            print("-" * 30)
            self.simulate_face_recognition(person)
            time.sleep(1)  # Pause between scenarios
        
        # Show final statistics
        self.show_statistics()

def main():
    """Run the demo"""
    print("🎪 Robot Face Recognition System - Demo Mode")
    print("This demo simulates the face recognition system without requiring a camera\n")
    
    demo = SimpleFaceDemo()
    demo.run_demo()
    
    print("\n✅ Demo completed!")
    print("\n🚀 To run the full system with camera:")
    print("   python robot_face_system.py")
    print("\n🔧 Features demonstrated:")
    print("   ✅ Unknown face detection and name asking")
    print("   ✅ Face database storage and retrieval")
    print("   ✅ Personalized greetings based on encounter history") 
    print("   ✅ Adaptive conversation based on familiarity")
    print("   ✅ Statistics and analytics tracking")

if __name__ == "__main__":
    main()
