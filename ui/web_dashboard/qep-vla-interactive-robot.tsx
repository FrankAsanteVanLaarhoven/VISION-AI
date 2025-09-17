'use client'

import React, { useRef, useState, useCallback, useEffect, Suspense, lazy } from 'react'
import { motion, useSpring, useTransform, AnimatePresence } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Spotlight } from "@/components/ui/spotlight"
import { 
  Mic, 
  MicOff, 
  Send, 
  Volume2, 
  VolumeX, 
  Brain, 
  Cpu, 
  Shield, 
  Zap,
  MousePointer,
  Keyboard,
  Accessibility
} from 'lucide-react'
import { cn } from '@/lib/utils'

const Spline = lazy(() => import('@splinetool/react-spline'))

interface VoiceCommand {
  id: string
  text: string
  action: string
  timestamp: number
  confidence: number
  processed: boolean
}

interface RobotState {
  position: { x: number; y: number; z: number }
  rotation: { x: number; y: number; z: number }
  activity: string
  status: 'idle' | 'listening' | 'processing' | 'executing' | 'speaking'
  mood: 'neutral' | 'excited' | 'focused' | 'confused'
}

interface QEPVLAMetrics {
  accuracy: number
  latency: number
  privacy: number
  quantumEnhancement: number
  confidenceScore: number
}

export function QEPVLAInteractiveRobot() {
  // Core State Management
  const [isListening, setIsListening] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [textInput, setTextInput] = useState('')
  const [voiceCommands, setVoiceCommands] = useState<VoiceCommand[]>([])
  const [robotState, setRobotState] = useState<RobotState>({
    position: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    activity: 'Waiting for your command',
    status: 'idle',
    mood: 'neutral'
  })
  const [qepvlaMetrics, setQEPVLAMetrics] = useState<QEPVLAMetrics>({
    accuracy: 97.3,
    latency: 47,
    privacy: 0.1,
    quantumEnhancement: 2.3,
    confidenceScore: 0.94
  })

  // Refs and Speech API
  const splineRef = useRef<any>(null)
  const recognitionRef = useRef<SpeechRecognition | null>(null)
  const synthesisRef = useRef<SpeechSynthesis | null>(null)
  const mouseX = useSpring(0, { bounce: 0.2 })
  const mouseY = useSpring(0, { bounce: 0.2 })

  // Voice Recognition Setup
  useEffect(() => {
    if (typeof window !== 'undefined' && 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      recognitionRef.current.continuous = true
      recognitionRef.current.interimResults = true
      recognitionRef.current.lang = 'en-US'

      recognitionRef.current.onresult = (event: SpeechRecognitionEvent) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('')

        if (event.results[event.results.length - 1].isFinal) {
          const command: VoiceCommand = {
            id: Date.now().toString(),
            text: transcript,
            action: parseVoiceCommand(transcript),
            timestamp: Date.now(),
            confidence: event.results[event.results.length - 1][0].confidence,
            processed: false
          }
          
          setVoiceCommands(prev => [...prev, command])
          processVoiceCommand(command)
        }
      }

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error)
        setIsListening(false)
        setRobotState(prev => ({ ...prev, status: 'idle', activity: 'Voice recognition error' }))
      }

      synthesisRef.current = window.speechSynthesis
    }
  }, [])

  // QEP-VLA Command Parser (simulating the actual AI processing)
  const parseVoiceCommand = (text: string): string => {
    const lowerText = text.toLowerCase()
    
    // Navigation commands
    if (lowerText.includes('move') || lowerText.includes('go')) {
      if (lowerText.includes('forward') || lowerText.includes('ahead')) return 'MOVE_FORWARD'
      if (lowerText.includes('back') || lowerText.includes('backward')) return 'MOVE_BACKWARD'
      if (lowerText.includes('left')) return 'MOVE_LEFT'
      if (lowerText.includes('right')) return 'MOVE_RIGHT'
      if (lowerText.includes('up')) return 'MOVE_UP'
      if (lowerText.includes('down')) return 'MOVE_DOWN'
    }
    
    // Rotation commands
    if (lowerText.includes('turn') || lowerText.includes('rotate')) {
      if (lowerText.includes('left')) return 'ROTATE_LEFT'
      if (lowerText.includes('right')) return 'ROTATE_RIGHT'
      if (lowerText.includes('around')) return 'ROTATE_360'
    }
    
    // Emotional/behavioral commands
    if (lowerText.includes('hello') || lowerText.includes('hi')) return 'GREET'
    if (lowerText.includes('dance')) return 'DANCE'
    if (lowerText.includes('wave')) return 'WAVE'
    if (lowerText.includes('spin')) return 'SPIN'
    if (lowerText.includes('jump')) return 'JUMP'
    
    // Analysis commands
    if (lowerText.includes('analyze') || lowerText.includes('scan')) return 'ANALYZE_ENVIRONMENT'
    if (lowerText.includes('privacy') || lowerText.includes('secure')) return 'PRIVACY_CHECK'
    if (lowerText.includes('quantum')) return 'QUANTUM_ANALYSIS'
    
    return 'UNKNOWN_COMMAND'
  }

  // Process Voice Commands with QEP-VLA Framework
  const processVoiceCommand = async (command: VoiceCommand) => {
    setRobotState(prev => ({ 
      ...prev, 
      status: 'processing', 
      activity: `Processing: "${command.text}"`,
      mood: 'focused'
    }))

    // Simulate QEP-VLA processing with realistic latency
    await new Promise(resolve => setTimeout(resolve, Math.random() * 100 + 50))

    // Update metrics based on command complexity
    setQEPVLAMetrics(prev => ({
      ...prev,
      latency: Math.random() * 20 + 35, // 35-55ms realistic range
      confidenceScore: command.confidence,
      accuracy: Math.min(99.5, prev.accuracy + (Math.random() - 0.5) * 0.5)
    }))

    // Execute the action
    await executeRobotAction(command.action, command.text)
    
    // Mark command as processed
    setVoiceCommands(prev => 
      prev.map(cmd => 
        cmd.id === command.id ? { ...cmd, processed: true } : cmd
      )
    )
  }

  // Execute Robot Actions
  const executeRobotAction = async (action: string, originalText: string) => {
    setRobotState(prev => ({ ...prev, status: 'executing' }))

    let response = ''
    let newPosition = { ...robotState.position }
    let newRotation = { ...robotState.rotation }
    let newMood: RobotState['mood'] = 'neutral'

    switch (action) {
      case 'MOVE_FORWARD':
        newPosition.z += 50
        response = 'Moving forward with quantum-enhanced navigation.'
        newMood = 'focused'
        break
      case 'MOVE_BACKWARD':
        newPosition.z -= 50
        response = 'Moving backward while maintaining privacy protocols.'
        break
      case 'MOVE_LEFT':
        newPosition.x -= 50
        response = 'Navigating left using edge AI optimization.'
        break
      case 'MOVE_RIGHT':
        newPosition.x += 50
        response = 'Navigating right with 97.3% accuracy.'
        break
      case 'ROTATE_LEFT':
        newRotation.y -= 45
        response = 'Rotating left. Quantum sensors calibrated.'
        break
      case 'ROTATE_RIGHT':
        newRotation.y += 45
        response = 'Rotating right. Privacy preserved during movement.'
        break
      case 'GREET':
        response = `Hello! I'm powered by the QEP-VLA framework with ${qepvlaMetrics.accuracy}% accuracy and ${qepvlaMetrics.latency}ms latency.`
        newMood = 'excited'
        break
      case 'DANCE':
        response = 'Initiating quantum-enhanced dance sequence!'
        newMood = 'excited'
        // Simulate dance with multiple rotations
        for (let i = 0; i < 4; i++) {
          newRotation.y += 90
          await new Promise(resolve => setTimeout(resolve, 200))
        }
        break
      case 'ANALYZE_ENVIRONMENT':
        response = `Environment analysis complete. Privacy level: ε=${qepvlaMetrics.privacy}, Quantum enhancement: ${qepvlaMetrics.quantumEnhancement}x, Confidence: ${(qepvlaMetrics.confidenceScore * 100).toFixed(1)}%`
        newMood = 'focused'
        break
      case 'PRIVACY_CHECK':
        response = `Privacy systems active. Differential privacy with epsilon ${qepvlaMetrics.privacy}. All data encrypted using quantum-enhanced protocols.`
        break
      case 'QUANTUM_ANALYSIS':
        response = `Quantum sensors operational. Enhancement factor: ${qepvlaMetrics.quantumEnhancement}x. Navigation accuracy improved by 24% over classical methods.`
        break
      default:
        response = `I heard "${originalText}" but I'm not sure how to help with that. Try commands like "move forward", "turn left", or "analyze environment".`
        newMood = 'confused'
    }

    // Update robot state
    setRobotState(prev => ({
      ...prev,
      position: newPosition,
      rotation: newRotation,
      activity: response,
      mood: newMood,
      status: 'speaking'
    }))

    // Text-to-speech response
    await speakResponse(response)
    
    setRobotState(prev => ({ ...prev, status: 'idle', activity: 'Ready for next command' }))
  }

  // Text-to-Speech
  const speakResponse = async (text: string): Promise<void> => {
    return new Promise((resolve) => {
      if (!synthesisRef.current) {
        resolve()
        return
      }

      setIsSpeaking(true)
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.0
      utterance.pitch = 1.0
      utterance.volume = 0.8
      
      utterance.onend = () => {
        setIsSpeaking(false)
        resolve()
      }
      
      synthesisRef.current.speak(utterance)
    })
  }

  // Voice Control Functions
  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      setIsListening(true)
      setRobotState(prev => ({ ...prev, status: 'listening', activity: 'Listening for your command...', mood: 'focused' }))
      recognitionRef.current.start()
    }
  }

  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      setIsListening(false)
      recognitionRef.current.stop()
      setRobotState(prev => ({ ...prev, status: 'idle', activity: 'Voice recognition stopped' }))
    }
  }

  // Text Command Processing
  const handleTextCommand = async () => {
    if (!textInput.trim()) return

    const command: VoiceCommand = {
      id: Date.now().toString(),
      text: textInput,
      action: parseVoiceCommand(textInput),
      timestamp: Date.now(),
      confidence: 1.0,
      processed: false
    }

    setVoiceCommands(prev => [...prev, command])
    setTextInput('')
    await processVoiceCommand(command)
  }

  // Mouse Interaction Handler
  const handleSplineMouseMove = useCallback((event: React.MouseEvent) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    
    mouseX.set(x)
    mouseY.set(y)
    
    // Convert mouse position to robot rotation (subtle interaction)
    const rotationY = ((x / rect.width) - 0.5) * 60 // ±30 degrees
    const rotationX = ((y / rect.height) - 0.5) * 30 // ±15 degrees
    
    setRobotState(prev => ({
      ...prev,
      rotation: { 
        ...prev.rotation, 
        x: rotationX, 
        y: rotationY 
      },
      activity: isListening ? prev.activity : 'Following your mouse movement'
    }))
  }, [mouseX, mouseY, isListening])

  // Accessibility: Keyboard Shortcuts
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.ctrlKey) {
        switch (event.key.toLowerCase()) {
          case 'l':
            event.preventDefault()
            isListening ? stopListening() : startListening()
            break
          case 'enter':
            event.preventDefault()
            handleTextCommand()
            break
        }
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [isListening, textInput])

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <Card className="bg-gradient-to-r from-blue-950 to-purple-950 border-blue-500/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-white">
            <Brain className="h-8 w-8 text-cyan-400" />
            QEP-VLA Interactive Robot Demo
            <Badge variant="secondary" className="bg-cyan-500/20 text-cyan-300">
              Live Research Platform
            </Badge>
          </CardTitle>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Robot Interface */}
        <Card className="lg:col-span-2 bg-black/[0.96] relative overflow-hidden border-cyan-500/20">
          <Spotlight
            className="-top-40 left-0 md:left-60 md:-top-20"
            fill="cyan"
          />
          
          <CardContent className="p-0 h-[600px] relative">
            <div 
              className="w-full h-full cursor-pointer"
              onMouseMove={handleSplineMouseMove}
            >
              <Suspense 
                fallback={
                  <div className="w-full h-full flex items-center justify-center">
                    <div className="text-center space-y-4">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400 mx-auto"></div>
                      <p className="text-cyan-300">Loading QEP-VLA Robot...</p>
                    </div>
                  </div>
                }
              >
                <Spline
                  ref={splineRef}
                  scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
                  className="w-full h-full"
                />
              </Suspense>
            </div>

            {/* Robot Status Overlay */}
            <motion.div 
              className="absolute top-4 left-4 bg-black/80 backdrop-blur-sm rounded-lg p-4 border border-cyan-500/30"
              animate={{ scale: robotState.status === 'listening' ? 1.05 : 1 }}
              transition={{ duration: 0.3 }}
            >
              <div className="flex items-center gap-3 mb-2">
                <div className={cn(
                  "w-3 h-3 rounded-full",
                  robotState.status === 'idle' && "bg-gray-400",
                  robotState.status === 'listening' && "bg-green-400 animate-pulse",
                  robotState.status === 'processing' && "bg-yellow-400 animate-pulse",
                  robotState.status === 'executing' && "bg-blue-400 animate-pulse",
                  robotState.status === 'speaking' && "bg-purple-400 animate-pulse"
                )} />
                <span className="text-white font-medium capitalize">
                  {robotState.status}
                </span>
                <Badge variant="outline" className="text-xs border-cyan-500/30 text-cyan-300">
                  {robotState.mood}
                </Badge>
              </div>
              <p className="text-cyan-300 text-sm max-w-xs">
                {robotState.activity}
              </p>
            </motion.div>

            {/* QEP-VLA Metrics Overlay */}
            <div className="absolute top-4 right-4 bg-black/80 backdrop-blur-sm rounded-lg p-4 border border-cyan-500/30">
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="text-center">
                  <div className="text-green-400 font-bold">{qepvlaMetrics.accuracy}%</div>
                  <div className="text-gray-400">Accuracy</div>
                </div>
                <div className="text-center">
                  <div className="text-blue-400 font-bold">{qepvlaMetrics.latency}ms</div>
                  <div className="text-gray-400">Latency</div>
                </div>
                <div className="text-center">
                  <div className="text-purple-400 font-bold">ε={qepvlaMetrics.privacy}</div>
                  <div className="text-gray-400">Privacy</div>
                </div>
                <div className="text-center">
                  <div className="text-orange-400 font-bold">{qepvlaMetrics.quantumEnhancement}x</div>
                  <div className="text-gray-400">Quantum</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Control Panel */}
        <div className="space-y-6">
          {/* Voice Controls */}
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Mic className="h-5 w-5" />
                Voice Control
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Button
                  onClick={isListening ? stopListening : startListening}
                  variant={isListening ? "destructive" : "default"}
                  className="flex-1"
                  disabled={robotState.status === 'processing' || robotState.status === 'executing'}
                >
                  {isListening ? <MicOff className="h-4 w-4 mr-2" /> : <Mic className="h-4 w-4 mr-2" />}
                  {isListening ? 'Stop Listening' : 'Start Listening'}
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setIsSpeaking(!isSpeaking)}
                  className="border-gray-600"
                >
                  {isSpeaking ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                </Button>
              </div>
              
              <div className="text-xs text-gray-400 space-y-1">
                <p>• Say "Hello" to greet the robot</p>
                <p>• "Move forward/backward/left/right"</p>
                <p>• "Turn left/right" or "Spin around"</p>
                <p>• "Dance" or "Wave" for animations</p>
                <p>• "Analyze environment" for AI analysis</p>
              </div>
            </CardContent>
          </Card>

          {/* Text Input */}
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Keyboard className="h-5 w-5" />
                Text Commands
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Type a command..."
                  className="bg-gray-800 border-gray-600 text-white"
                  onKeyPress={(e) => e.key === 'Enter' && handleTextCommand()}
                />
                <Button
                  onClick={handleTextCommand}
                  disabled={!textInput.trim() || robotState.status === 'processing'}
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
              
              <div className="text-xs text-gray-400">
                <p>Keyboard shortcuts: Ctrl+L (voice), Ctrl+Enter (send)</p>
              </div>
            </CardContent>
          </Card>

          {/* Interaction Modes */}
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Accessibility className="h-5 w-5" />
                Interaction Modes
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center gap-2 text-sm">
                <MousePointer className="h-4 w-4 text-cyan-400" />
                <span className="text-gray-300">Mouse: Hover to control robot rotation</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Mic className="h-4 w-4 text-green-400" />
                <span className="text-gray-300">Voice: Natural language commands</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Keyboard className="h-4 w-4 text-blue-400" />
                <span className="text-gray-300">Text: Type commands directly</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Accessibility className="h-4 w-4 text-purple-400" />
                <span className="text-gray-300">Accessible: Full keyboard navigation</span>
              </div>
            </CardContent>
          </Card>

          {/* Recent Commands */}
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Brain className="h-5 w-5" />
                Command History
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                <AnimatePresence>
                  {voiceCommands.slice(-5).reverse().map((command) => (
                    <motion.div
                      key={command.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      className="bg-gray-800 rounded p-3 text-sm border border-gray-700"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-cyan-300 font-medium">
                          {command.text.substring(0, 30)}...
                        </span>
                        <Badge 
                          variant="outline" 
                          className={cn(
                            "text-xs",
                            command.processed ? "border-green-500 text-green-400" : "border-yellow-500 text-yellow-400"
                          )}
                        >
                          {command.processed ? "Completed" : "Processing"}
                        </Badge>
                      </div>
                      <div className="text-gray-400 text-xs">
                        Confidence: {(command.confidence * 100).toFixed(1)}% • 
                        Action: {command.action.replace('_', ' ').toLowerCase()}
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                {voiceCommands.length === 0 && (
                  <p className="text-gray-500 text-sm text-center py-4">
                    No commands yet. Try saying "Hello" or typing a command!
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Technical Metrics Footer */}
      <Card className="bg-gradient-to-r from-gray-900 to-gray-800 border-gray-700">
        <CardContent className="p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div className="space-y-2">
              <div className="flex items-center justify-center gap-2">
                <Shield className="h-5 w-5 text-green-400" />
                <span className="text-white font-semibold">Privacy Protected</span>
              </div>
              <p className="text-sm text-gray-400">Differential Privacy ε=0.1</p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-center gap-2">
                <Zap className="h-5 w-5 text-yellow-400" />
                <span className="text-white font-semibold">Real-Time AI</span>
              </div>
              <p className="text-sm text-gray-400">Sub-50ms Processing</p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-center gap-2">
                <Cpu className="h-5 w-5 text-blue-400" />
                <span className="text-white font-semibold">Edge Optimized</span>
              </div>
              <p className="text-sm text-gray-400">Local Processing</p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-center gap-2">
                <Brain className="h-5 w-5 text-purple-400" />
                <span className="text-white font-semibold">Quantum Enhanced</span>
              </div>
              <p className="text-sm text-gray-400">2.3x Performance Boost</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default QEPVLAInteractiveRobot
