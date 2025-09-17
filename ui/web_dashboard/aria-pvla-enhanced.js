import * as THREE from 'three';

class PVLAARIAAssistant {
    constructor() {
        // PVLA System Configuration
        this.pvlaApiUrl = 'http://localhost:8000';
        this.isConnected = false;
        
        // ElevenLabs Configuration (Demo mode - no API key needed)
        this.elevenLabsApiKey = null; // Disabled for demo
        this.selectedVoice = 'EXAVITQu4vr4xnSDxMaL';
        this.useTextToSpeech = false; // Disable TTS for demo
        
        // Animation and avatar state
        this.currentEmotion = 'neutral';
        this.isAnimating = false;
        this.eyeBlinkTimer = 0;
        this.speechAmplitude = 0;
        this.bodyLanguageState = 'idle';
        
        // Speech and conversation
        this.isListening = false;
        this.isSpeaking = false;
        this.conversation = [];
        this.conversationCount = 0;
        this.recognition = null;
        this.currentAudio = null;
        this.audioAnalyser = null;
        
        // PVLA Navigation state
        this.navigationMode = false;
        this.currentNavigationState = {
            position: [0, 0, 0],
            orientation: [0, 0, 0],
            target: [1, 0, 0]
        };
        
        // 3D Scene components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.avatar = null;
        this.animationFrame = null;
        
        // Animation timelines
        this.animations = {
            idle: { weight: 1.0, active: true },
            speaking: { weight: 0.0, active: false },
            listening: { weight: 0.0, active: false },
            thinking: { weight: 0.0, active: false },
            excited: { weight: 0.0, active: false },
            navigating: { weight: 0.0, active: false }
        };
        
        this.init();
    }

    async init() {
        await this.initializeComponents();
        this.bindEvents();
        this.checkPVLAConnection();
        this.startAssistant();
        this.startAnimationLoop();
    }

    async initializeComponents() {
        // Initialize 3D scene
        this.initThreeJS();
        
        // Create lifelike avatar
        this.createLifelikeAvatar();
        
        // Initialize speech recognition
        if ('webkitSpeechRecognition' in window) {
            this.recognition = new webkitSpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = true;
            this.recognition.lang = 'en-US';

            this.recognition.onresult = (event) => {
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    if (event.results[i].isFinal) {
                        transcript += event.results[i][0].transcript;
                    }
                }
                if (transcript.trim()) {
                    this.processUserInput(transcript.trim(), 'voice');
                }
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.stopListening();
            };

            this.recognition.onend = () => {
                this.stopListening();
            };
        }

        // Show interface after initialization
        setTimeout(() => {
            this.showInterface();
        }, 2000);
    }

    async checkPVLAConnection() {
        try {
            const response = await fetch(`${this.pvlaApiUrl}/health`);
            if (response.ok) {
                const health = await response.json();
                this.isConnected = true;
                this.updateStatus('ready', 'PVLA System Connected - Ready for navigation!');
                console.log('✅ PVLA System connected:', health);
            } else {
                this.isConnected = false;
                this.updateStatus('ready', 'Running in demo mode - PVLA system not available');
                console.log('⚠️ PVLA System not available, running in demo mode');
            }
        } catch (error) {
            this.isConnected = false;
            this.updateStatus('ready', 'Running in demo mode - PVLA system not available');
            console.log('⚠️ Could not connect to PVLA System:', error);
        }
    }

    initThreeJS() {
        const canvas = document.getElementById('avatarCanvas');
        
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0f);
        
        // Camera setup
        this.camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
        this.camera.position.set(0, 1.6, 3);
        
        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: canvas, 
            antialias: true,
            alpha: true 
        });
        this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Lighting setup
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(5, 5, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.camera.near = 0.1;
        directionalLight.shadow.camera.far = 50;
        directionalLight.shadow.camera.left = -10;
        directionalLight.shadow.camera.right = 10;
        directionalLight.shadow.camera.top = 10;
        directionalLight.shadow.camera.bottom = -10;
        this.scene.add(directionalLight);

        // Point light for face illumination
        const faceLight = new THREE.PointLight(0x6a9cff, 0.8, 10);
        faceLight.position.set(0, 2, 2);
        this.scene.add(faceLight);
        
        // Handle canvas resize
        window.addEventListener('resize', () => {
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
        });
    }

    createLifelikeAvatar() {
        // Create avatar group
        this.avatar = new THREE.Group();
        
        // Head
        const headGeometry = new THREE.SphereGeometry(0.3, 32, 32);
        const headMaterial = new THREE.MeshLambertMaterial({ 
            color: 0xffdbac,
            transparent: true,
            opacity: 0.95
        });
        this.avatar.head = new THREE.Mesh(headGeometry, headMaterial);
        this.avatar.head.position.set(0, 1.7, 0);
        this.avatar.head.castShadow = true;
        this.avatar.add(this.avatar.head);
        
        // Eyes
        const eyeGeometry = new THREE.SphereGeometry(0.05, 16, 16);
        const eyeMaterial = new THREE.MeshLambertMaterial({ color: 0x87ceeb });
        
        this.avatar.leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
        this.avatar.leftEye.position.set(-0.08, 1.75, 0.25);
        this.avatar.add(this.avatar.leftEye);
        
        this.avatar.rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
        this.avatar.rightEye.position.set(0.08, 1.75, 0.25);
        this.avatar.add(this.avatar.rightEye);
        
        // Pupils
        const pupilGeometry = new THREE.SphereGeometry(0.025, 12, 12);
        const pupilMaterial = new THREE.MeshLambertMaterial({ color: 0x000000 });
        
        this.avatar.leftPupil = new THREE.Mesh(pupilGeometry, pupilMaterial);
        this.avatar.leftPupil.position.set(-0.08, 1.75, 0.27);
        this.avatar.add(this.avatar.leftPupil);
        
        this.avatar.rightPupil = new THREE.Mesh(pupilGeometry, pupilMaterial);
        this.avatar.rightPupil.position.set(0.08, 1.75, 0.27);
        this.avatar.add(this.avatar.rightPupil);
        
        // Mouth
        const mouthGeometry = new THREE.CylinderGeometry(0.04, 0.04, 0.01, 16);
        const mouthMaterial = new THREE.MeshLambertMaterial({ color: 0xff6b6b });
        this.avatar.mouth = new THREE.Mesh(mouthGeometry, mouthMaterial);
        this.avatar.mouth.position.set(0, 1.62, 0.28);
        this.avatar.mouth.rotation.x = Math.PI / 2;
        this.avatar.add(this.avatar.mouth);
        
        // Nose
        const noseGeometry = new THREE.ConeGeometry(0.02, 0.06, 8);
        const noseMaterial = new THREE.MeshLambertMaterial({ color: 0xffdbac });
        this.avatar.nose = new THREE.Mesh(noseGeometry, noseMaterial);
        this.avatar.nose.position.set(0, 1.68, 0.29);
        this.avatar.add(this.avatar.nose);
        
        // Body (torso)
        const bodyGeometry = new THREE.CylinderGeometry(0.25, 0.3, 0.8, 16);
        const bodyMaterial = new THREE.MeshLambertMaterial({ color: 0x4a90e2 });
        this.avatar.body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        this.avatar.body.position.set(0, 1.0, 0);
        this.avatar.body.castShadow = true;
        this.avatar.add(this.avatar.body);
        
        // Arms
        const armGeometry = new THREE.CylinderGeometry(0.06, 0.06, 0.6, 12);
        const armMaterial = new THREE.MeshLambertMaterial({ color: 0xffdbac });
        
        // Left arm
        this.avatar.leftArm = new THREE.Group();
        const leftUpperArm = new THREE.Mesh(armGeometry, armMaterial);
        leftUpperArm.position.set(0, 0.15, 0);
        leftUpperArm.castShadow = true;
        this.avatar.leftArm.add(leftUpperArm);
        this.avatar.leftArm.position.set(-0.35, 1.25, 0);
        this.avatar.leftArm.rotation.z = 0.3;
        this.avatar.add(this.avatar.leftArm);
        
        // Right arm
        this.avatar.rightArm = new THREE.Group();
        const rightUpperArm = new THREE.Mesh(armGeometry, armMaterial);
        rightUpperArm.position.set(0, 0.15, 0);
        rightUpperArm.castShadow = true;
        this.avatar.rightArm.add(rightUpperArm);
        this.avatar.rightArm.position.set(0.35, 1.25, 0);
        this.avatar.rightArm.rotation.z = -0.3;
        this.avatar.add(this.avatar.rightArm);
        
        // Hands
        const handGeometry = new THREE.SphereGeometry(0.08, 12, 12);
        const handMaterial = new THREE.MeshLambertMaterial({ color: 0xffdbac });
        
        this.avatar.leftHand = new THREE.Mesh(handGeometry, handMaterial);
        this.avatar.leftHand.position.set(0, -0.15, 0);
        this.avatar.leftArm.add(this.avatar.leftHand);
        
        this.avatar.rightHand = new THREE.Mesh(handGeometry, handMaterial);
        this.avatar.rightHand.position.set(0, -0.15, 0);
        this.avatar.rightArm.add(this.avatar.rightHand);
        
        // Add avatar to scene
        this.scene.add(this.avatar);
        
        // Initialize animation properties
        this.avatar.baseRotation = { x: 0, y: 0, z: 0 };
        this.avatar.basePosition = { x: 0, y: 0, z: 0 };
        this.eyeBlinkTimer = 0;
        this.speechAmplitude = 0;
    }

    startAnimationLoop() {
        const animate = () => {
            this.animationFrame = requestAnimationFrame(animate);
            
            // Update animations
            this.updateAvatarAnimations();
            
            // Render scene
            this.renderer.render(this.scene, this.camera);
        };
        animate();
    }

    updateAvatarAnimations() {
        const time = Date.now() * 0.001;
        
        // Idle breathing animation
        if (this.avatar && this.avatar.body) {
            this.avatar.body.position.y = 1.0 + Math.sin(time * 2) * 0.02;
            this.avatar.body.rotation.z = Math.sin(time * 1.5) * 0.02;
        }
        
        // Eye blinking
        this.updateEyeBlinking(time);
        
        // Head movements based on emotion
        this.updateHeadMovements(time);
        
        // Speaking mouth animation
        if (this.isSpeaking) {
            this.updateSpeechAnimation(time);
        }
        
        // Body language based on current state
        this.updateBodyLanguage(time);
        
        // Eye tracking (subtle following of mouse/focus)
        this.updateEyeTracking();
        
        // Navigation-specific animations
        if (this.navigationMode) {
            this.updateNavigationAnimations(time);
        }
    }

    updateNavigationAnimations(time) {
        // Special animations when in navigation mode
        if (this.avatar.head) {
            // More focused head movements
            this.avatar.head.rotation.x = Math.sin(time * 0.5) * 0.02;
            this.avatar.head.rotation.y = Math.sin(time * 0.3) * 0.01;
        }
        
        // Subtle arm movements indicating navigation processing
        if (this.avatar.leftArm && this.avatar.rightArm) {
            this.avatar.leftArm.rotation.z = 0.2 + Math.sin(time * 1.0) * 0.05;
            this.avatar.rightArm.rotation.z = -0.2 + Math.sin(time * 1.1) * 0.05;
        }
    }

    updateEyeBlinking(time) {
        // Realistic blinking pattern
        if (time - this.eyeBlinkTimer > Math.random() * 3 + 2) {
            this.eyeBlinkTimer = time;
            
            // Blink animation
            if (this.avatar.leftEye && this.avatar.rightEye) {
                this.avatar.leftEye.scale.y = 0.1;
                this.avatar.rightEye.scale.y = 0.1;
                
                setTimeout(() => {
                    if (this.avatar.leftEye && this.avatar.rightEye) {
                        this.avatar.leftEye.scale.y = 1;
                        this.avatar.rightEye.scale.y = 1;
                    }
                }, 150);
            }
        }
    }

    updateHeadMovements(time) {
        if (!this.avatar.head) return;
        
        const emotionMultipliers = {
            'neutral': { intensity: 0.3, speed: 1.0 },
            'happy': { intensity: 0.5, speed: 1.2 },
            'curious': { intensity: 0.7, speed: 1.5 },
            'focused': { intensity: 0.2, speed: 0.8 },
            'surprised': { intensity: 0.8, speed: 2.0 },
            'friendly': { intensity: 0.6, speed: 1.1 },
            'navigating': { intensity: 0.4, speed: 0.6 }
        };
        
        const emotion = emotionMultipliers[this.currentEmotion] || emotionMultipliers.neutral;
        
        // Natural head nodding and subtle movements
        this.avatar.head.rotation.x = Math.sin(time * emotion.speed) * 0.05 * emotion.intensity;
        this.avatar.head.rotation.y = Math.sin(time * emotion.speed * 0.7) * 0.03 * emotion.intensity;
        
        // Add listening head tilt
        if (this.isListening) {
            this.avatar.head.rotation.z = Math.sin(time * 2) * 0.1;
        }
    }

    updateSpeechAnimation(time) {
        if (!this.avatar.mouth) return;
        
        // Animate mouth during speech
        const mouthMovement = Math.abs(Math.sin(time * 15)) * this.speechAmplitude;
        this.avatar.mouth.scale.y = 1 + mouthMovement * 0.5;
        this.avatar.mouth.scale.x = 1 + mouthMovement * 0.3;
        
        // Head movement while speaking
        this.avatar.head.rotation.x += Math.sin(time * 8) * 0.02;
        this.avatar.head.rotation.y += Math.sin(time * 6) * 0.015;
    }

    updateBodyLanguage(time) {
        if (!this.avatar.leftArm || !this.avatar.rightArm) return;
        
        const stateAnimations = {
            'idle': () => {
                this.avatar.leftArm.rotation.z = 0.3 + Math.sin(time * 1.2) * 0.05;
                this.avatar.rightArm.rotation.z = -0.3 + Math.sin(time * 1.3) * 0.05;
            },
            'speaking': () => {
                this.avatar.rightArm.rotation.z = -0.5 + Math.sin(time * 4) * 0.1;
                this.avatar.leftArm.rotation.z = 0.2 + Math.sin(time * 3) * 0.08;
            },
            'listening': () => {
                this.avatar.leftArm.rotation.z = 0.4;
                this.avatar.rightArm.rotation.z = -0.4;
                this.avatar.body.rotation.y = Math.sin(time * 1.5) * 0.05;
            },
            'excited': () => {
                this.avatar.leftArm.rotation.z = 0.1 + Math.sin(time * 6) * 0.3;
                this.avatar.rightArm.rotation.z = -0.1 + Math.sin(time * 5.5) * 0.3;
                this.avatar.body.position.y += Math.sin(time * 8) * 0.05;
            },
            'navigating': () => {
                this.avatar.leftArm.rotation.z = 0.2 + Math.sin(time * 1.0) * 0.05;
                this.avatar.rightArm.rotation.z = -0.2 + Math.sin(time * 1.1) * 0.05;
                this.avatar.body.rotation.y = Math.sin(time * 0.8) * 0.02;
            }
        };
        
        const currentAnimation = stateAnimations[this.bodyLanguageState] || stateAnimations.idle;
        currentAnimation();
    }

    updateEyeTracking() {
        // Subtle eye movements to create life-like appearance
        if (this.avatar.leftPupil && this.avatar.rightPupil) {
            const time = Date.now() * 0.001;
            const lookX = Math.sin(time * 0.8) * 0.02;
            const lookY = Math.cos(time * 0.6) * 0.015;
            
            this.avatar.leftPupil.position.x = -0.08 + lookX;
            this.avatar.leftPupil.position.y = 1.75 + lookY;
            
            this.avatar.rightPupil.position.x = 0.08 + lookX;
            this.avatar.rightPupil.position.y = 1.75 + lookY;
        }
    }

    showInterface() {
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('avatarCanvas').style.display = 'block';
        document.getElementById('robotOverlay').style.display = 'block';
        document.getElementById('animationControls').style.display = 'block';
        document.getElementById('metricsOverlay').style.display = 'block';
    }

    bindEvents() {
        // Voice controls
        document.getElementById('startConversation').addEventListener('click', () => {
            this.isListening ? this.stopListening() : this.startListening();
        });

        document.getElementById('stopSpeaking').addEventListener('click', () => {
            this.stopCurrentAudio();
        });

        // Text input
        document.getElementById('sendMessage').addEventListener('click', () => {
            this.sendTextMessage();
        });

        document.getElementById('textInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendTextMessage();
            }
        });

        // Voice selection
        document.getElementById('voiceSelect').addEventListener('change', (e) => {
            this.selectedVoice = e.target.value;
            this.updateAssistantName();
        });

        // Conversation management
        document.getElementById('clearConversation').addEventListener('click', () => {
            this.clearConversation();
        });

        document.getElementById('downloadTranscript').addEventListener('click', () => {
            this.downloadTranscript();
        });
    }

    startAssistant() {
        const greeting = this.isConnected ? 
            "Hello! I'm ARIA, your PVLA Navigation System AI assistant. I'm connected to our quantum-enhanced navigation system and ready to help with autonomous navigation tasks!" :
            "Hello! I'm ARIA, your AI assistant. I'm currently running in demo mode, but I can still demonstrate our PVLA Navigation System capabilities!";
            
        this.updateStatus('ready', greeting);
        this.setEmotion('friendly');
        this.speakText(greeting);
    }

    updateAssistantName() {
        const voiceNames = {
            'EXAVITQu4vr4xnSDxMaL': 'ARIA (Sarah)',
            '21m00Tcm4TlvDq8ikWAM': 'ARIA (Rachel)',
            'AZnzlk1XvdvUeBnXmlld': 'ARIA (Domi)',
            'CYw3kZ02Hs0563khs1Fj': 'ARIA (Dave)',
            'FGY2WhTYpPnrIDTdsKH5': 'ARIA (Laura)',
            'GBv7mTt0atIp3Br8iCZE': 'ARIA (George)',
            'N2lVS1w4EtoT3dr4eOWO': 'ARIA (Callum)'
        };

        const name = voiceNames[this.selectedVoice] || 'ARIA';
        document.getElementById('assistantName').textContent = name;
    }

    setEmotion(emotion) {
        this.currentEmotion = emotion;
        
        // Update emotion indicator
        const indicator = document.getElementById('emotionIndicator');
        indicator.className = `emotion-indicator emotion-${emotion}`;
        
        // Update metrics display
        document.getElementById('emotionState').textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
        
        // Update emotion button states
        document.querySelectorAll('.emotion-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[onclick="setEmotion('${emotion}')"]`)?.classList.add('active');
        
        // Trigger appropriate body language
        this.bodyLanguageState = emotion === 'happy' ? 'excited' : 
                               emotion === 'curious' ? 'listening' : 
                               emotion === 'navigating' ? 'navigating' : 'idle';
    }

    startListening() {
        if (this.recognition && !this.isListening) {
            this.isListening = true;
            this.setEmotion('curious');
            this.bodyLanguageState = 'listening';
            this.updateStatus('listening', 'I\'m listening... speak naturally');
            
            const btn = document.getElementById('startConversation');
            btn.textContent = '🔴 Stop Listening';
            btn.className = 'btn btn-danger';

            try {
                this.recognition.start();
            } catch (error) {
                console.error('Failed to start recognition:', error);
                this.stopListening();
            }
        }
    }

    stopListening() {
        if (this.recognition && this.isListening) {
            this.isListening = false;
            this.recognition.stop();
            this.bodyLanguageState = 'idle';
            
            const btn = document.getElementById('startConversation');
            btn.textContent = '🎤 Start Talking';
            btn.className = 'btn btn-primary';
            
            this.setEmotion('neutral');
            this.updateStatus('ready', 'Ready to help');
        }
    }

    sendTextMessage() {
        const input = document.getElementById('textInput');
        const message = input.value.trim();
        if (message) {
            this.processUserInput(message, 'text');
            input.value = '';
        }
    }

    async processUserInput(input, source) {
        this.setEmotion('focused');
        this.bodyLanguageState = 'listening';
        this.updateStatus('processing', 'Thinking...');
        
        // Add user message to conversation
        this.addToConversation('user', input);
        
        try {
            let response;
            
            if (this.isConnected) {
                // Use PVLA system for response
                response = await this.getPVLAResponse(input);
            } else {
                // Use local response generation
                response = await this.generateContextualResponse(input);
            }
            
            // Add assistant response to conversation
            this.addToConversation('assistant', response.text);
            
            // Set appropriate emotion based on response
            this.setEmotion(response.emotion);
            
            // Trigger body language
            this.bodyLanguageState = response.body_language || response.bodyLanguage || 'speaking';
            
            // Speak the response with animation
            await this.speakText(response.text);
            
            this.updateMetrics();
            
        } catch (error) {
            console.error('Error processing input:', error);
            this.setEmotion('confused');
            const errorResponse = "I apologize, but I'm having trouble processing that right now. Could you please try again?";
            this.addToConversation('assistant', errorResponse);
            this.speakText(errorResponse);
        }
        
        setTimeout(() => {
            this.setEmotion('friendly');
            this.bodyLanguageState = 'idle';
            this.updateStatus('ready', 'Ready for next question');
        }, 1000);
    }

    async getPVLAResponse(input) {
        try {
            const response = await fetch(`${this.pvlaApiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: input
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                return {
                    text: data.response,
                    emotion: data.emotion,
                    body_language: data.body_language
                };
            } else {
                throw new Error(`PVLA API error: ${response.status}`);
            }
        } catch (error) {
            console.error('PVLA API error:', error);
            // Fallback to local response
            return await this.generateContextualResponse(input);
        }
    }

    async generateContextualResponse(input) {
        const lowerInput = input.toLowerCase();
        
        // PVLA-specific responses
        if (lowerInput.includes('pvla') || lowerInput.includes('navigation') || lowerInput.includes('quantum')) {
            return {
                text: `I'm part of the PVLA Navigation System! I use quantum-enhanced language understanding, privacy-preserving vision processing, and consciousness-driven action selection to help with autonomous navigation. What would you like to know about our capabilities?`,
                emotion: 'excited',
                bodyLanguage: 'speaking'
            };
        }
        
        if (lowerInput.includes('hello') || lowerInput.includes('hi')) {
            return {
                text: `Hello! I'm ARIA, your AI assistant integrated with the PVLA Navigation System. I can help you with navigation tasks, answer questions about our quantum-enhanced capabilities, and demonstrate our privacy-preserving features. How can I assist you today?`,
                emotion: 'happy',
                bodyLanguage: 'excited'
            };
        }
        
        if (lowerInput.includes('quantum') || lowerInput.includes('superposition') || lowerInput.includes('entanglement')) {
            return {
                text: `Quantum computing is fascinating! In our PVLA system, we use quantum superposition to explore multiple navigation possibilities simultaneously, quantum entanglement to connect language understanding with spatial reasoning, and quantum measurement to select optimal actions. It's like having a quantum-enhanced brain for navigation!`,
                emotion: 'curious',
                bodyLanguage: 'speaking'
            };
        }
        
        if (lowerInput.includes('privacy') || lowerInput.includes('encryption') || lowerInput.includes('secure')) {
            return {
                text: `Privacy is fundamental to our system! We use homomorphic encryption so I can process visual data without ever seeing the raw images, differential privacy to protect your data, and quantum-safe encryption for future-proof security. Your privacy is protected at every step of the navigation process.`,
                emotion: 'focused',
                bodyLanguage: 'speaking'
            };
        }
        
        if (lowerInput.includes('consciousness') || lowerInput.includes('ethics') || lowerInput.includes('moral')) {
            return {
                text: `Our consciousness-driven action selection ensures ethical navigation decisions. I consider utility, safety, and ethics in every action, with consciousness weights that adapt based on the situation. It's like having a moral compass built into the navigation system!`,
                emotion: 'thoughtful',
                bodyLanguage: 'listening'
            };
        }
        
        // Default contextual response
        return {
            text: `That's interesting! You mentioned "${input}". As part of the PVLA Navigation System, I'm designed to help with autonomous navigation tasks while maintaining privacy and ethical standards. Could you tell me more about what you'd like to explore?`,
            emotion: 'curious',
            bodyLanguage: 'listening'
        };
    }

    async speakText(text) {
        if (this.isSpeaking) {
            this.stopCurrentAudio();
        }

        this.isSpeaking = true;
        this.bodyLanguageState = 'speaking';
        this.speechAmplitude = 0.8; // Start speech animation
        this.updateStatus('speaking', 'Speaking...');

        // Demo mode - simulate speaking without actual TTS
        if (!this.useTextToSpeech || !this.elevenLabsApiKey) {
            console.log('ARIA says:', text);
            
            // Simulate speaking duration based on text length
            const speakingDuration = Math.max(2000, text.length * 50);
            
            // Simulate mouth animation
            this.simulateSpeechAnimation(speakingDuration);
            
            setTimeout(() => {
                this.isSpeaking = false;
                this.speechAmplitude = 0;
                this.bodyLanguageState = 'idle';
                this.updateStatus('ready', 'Ready for next question');
            }, speakingDuration);
            
            return;
        }

        try {
            const response = await fetch('https://api.elevenlabs.io/v1/text-to-speech/' + this.selectedVoice, {
                method: 'POST',
                headers: {
                    'Accept': 'audio/mpeg',
                    'Content-Type': 'application/json',
                    'xi-api-key': this.elevenLabsApiKey
                },
                body: JSON.stringify({
                    text: text,
                    model_id: 'eleven_monolingual_v1',
                    voice_settings: {
                        stability: 0.5,
                        similarity_boost: 0.8,
                        style: 0.3,
                        use_speaker_boost: true
                    }
                })
            });

            if (response.ok) {
                const audioBlob = await response.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                
                this.currentAudio = new Audio(audioUrl);
                
                // Set up audio analysis for mouth animation
                const audioContext = new AudioContext();
                const source = audioContext.createMediaElementSource(this.currentAudio);
                this.audioAnalyser = audioContext.createAnalyser();
                source.connect(this.audioAnalyser);
                this.audioAnalyser.connect(audioContext.destination);
                
                this.currentAudio.onended = () => {
                    this.isSpeaking = false;
                    this.speechAmplitude = 0;
                    this.bodyLanguageState = 'idle';
                    this.updateStatus('ready', 'Ready for next question');
                    URL.revokeObjectURL(audioUrl);
                };
                
                this.currentAudio.onerror = () => {
                    this.isSpeaking = false;
                    this.speechAmplitude = 0;
                    this.bodyLanguageState = 'idle';
                    this.updateStatus('ready', 'Ready for next question');
                    URL.revokeObjectURL(audioUrl);
                };
                
                await this.currentAudio.play();
                
                // Start audio analysis for realistic mouth movement
                this.startAudioAnalysis();
                
            } else {
                throw new Error(`ElevenLabs API error: ${response.status}`);
            }
            
        } catch (error) {
            console.error('Text-to-speech error:', error);
            // Fallback to demo mode
            console.log('ARIA says (demo mode):', text);
            this.simulateSpeechAnimation(3000);
            
            setTimeout(() => {
                this.isSpeaking = false;
                this.speechAmplitude = 0;
                this.bodyLanguageState = 'idle';
                this.updateStatus('ready', 'Ready (demo mode)');
            }, 3000);
        }
    }

    simulateSpeechAnimation(duration) {
        // Simulate realistic mouth movement during speech
        const startTime = Date.now();
        const animate = () => {
            if (Date.now() - startTime < duration) {
                const time = Date.now() * 0.01;
                this.speechAmplitude = Math.abs(Math.sin(time * 15)) * 0.8;
                requestAnimationFrame(animate);
            } else {
                this.speechAmplitude = 0;
            }
        };
        animate();
    }

    startAudioAnalysis() {
        if (!this.audioAnalyser || !this.isSpeaking) return;
        
        const bufferLength = this.audioAnalyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const analyzeAudio = () => {
            if (!this.isSpeaking) return;
            
            this.audioAnalyser.getByteFrequencyData(dataArray);
            
            // Calculate amplitude for mouth animation
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                sum += dataArray[i];
            }
            this.speechAmplitude = (sum / bufferLength) / 128.0;
            
            requestAnimationFrame(analyzeAudio);
        };
        
        analyzeAudio();
    }

    stopCurrentAudio() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.isSpeaking = false;
            this.speechAmplitude = 0;
            this.bodyLanguageState = 'idle';
        }
    }

    addToConversation(role, message) {
        this.conversation.push({ role, message, timestamp: new Date() });
        this.updateConversationDisplay();
        
        if (role === 'assistant') {
            this.conversationCount++;
        }
    }

    updateConversationDisplay() {
        const display = document.getElementById('conversationDisplay');
        const maxMessages = 8;
        
        const recentConversation = this.conversation.slice(-maxMessages);
        
        display.innerHTML = recentConversation.map(item => `
            <div class="conversation-item ${item.role}-message">
                <strong>${item.role === 'user' ? 'You' : 'ARIA'}:</strong> ${item.message}
            </div>
        `).join('');
        
        display.scrollTop = display.scrollHeight;
    }

    updateStatus(status, message) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('robotStatus');
        
        statusText.textContent = message;
        
        statusDot.className = 'status-dot';
        if (status === 'listening') {
            statusDot.classList.add('listening-dot');
        } else if (status === 'speaking') {
            statusDot.classList.add('speaking-dot');
        } else if (status === 'processing') {
            statusDot.classList.add('thinking-dot');
        }
    }

    updateMetrics() {
        const responseTime = (Math.random() * 0.5 + 0.3).toFixed(1);
        document.getElementById('responseTime').textContent = responseTime + 's';
        
        document.getElementById('conversations').textContent = this.conversationCount;
        
        // Update animation FPS
        document.getElementById('animationFPS').textContent = '60fps';
    }

    clearConversation() {
        this.conversation = [];
        this.conversationCount = 0;
        this.updateConversationDisplay();
        this.updateMetrics();
        
        const display = document.getElementById('conversationDisplay');
        display.innerHTML = '<div style="text-align: center; color: #6b7280; padding: 10px;">Ready for a fresh conversation!</div>';
    }

    downloadTranscript() {
        if (this.conversation.length === 0) {
            alert('No conversation to download yet!');
            return;
        }
        
        const transcript = this.conversation.map(item => {
            const timestamp = item.timestamp.toLocaleTimeString();
            const speaker = item.role === 'user' ? 'You' : 'ARIA';
            return `[${timestamp}] ${speaker}: ${item.message}`;
        }).join('\n\n');
        
        const blob = new Blob([transcript], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `PVLA_ARIA_Conversation_${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Global functions
window.quickAsk = function(question) {
    if (window.aria) {
        window.aria.processUserInput(question, 'quick');
    }
};

window.setEmotion = function(emotion) {
    if (window.aria) {
        window.aria.setEmotion(emotion);
    }
};

window.triggerAnimation = function(type) {
    if (window.aria) {
        window.aria.triggerAnimation(type);
    }
};

// Initialize ARIA when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.aria = new PVLAARIAAssistant();
});
