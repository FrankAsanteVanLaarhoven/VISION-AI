// components/VoiceProcessor.tsx
'use client'

import { useRef, useCallback, useState } from 'react'

interface VoiceProcessorOptions {
  language?: string
  continuous?: boolean
  interimResults?: boolean
  maxAlternatives?: number
}

export class AdvancedVoiceProcessor {
  private recognition: SpeechRecognition | null = null
  private synthesis: SpeechSynthesis | null = null
  private isListening = false
  private options: VoiceProcessorOptions

  constructor(options: VoiceProcessorOptions = {}) {
    this.options = {
      language: 'en-US',
      continuous: true,
      interimResults: true,
      maxAlternatives: 3,
      ...options
    }
    
    this.initializeSpeechRecognition()
    this.initializeSpeechSynthesis()
  }

  private initializeSpeechRecognition() {
    if (typeof window !== 'undefined' && 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition
      this.recognition = new SpeechRecognition()
      
      this.recognition.continuous = this.options.continuous!
      this.recognition.interimResults = this.options.interimResults!
      this.recognition.lang = this.options.language!
      this.recognition.maxAlternatives = this.options.maxAlternatives!
    }
  }

  private initializeSpeechSynthesis() {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      this.synthesis = window.speechSynthesis
    }
  }

  startListening(
    onResult: (transcript: string, confidence: number, isFinal: boolean) => void,
    onError: (error: string) => void
  ) {
    if (!this.recognition || this.isListening) return

    this.recognition.onresult = (event: SpeechRecognitionEvent) => {
      const result = event.results[event.results.length - 1]
      const transcript = result[0].transcript
      const confidence = result[0].confidence
      const isFinal = result.isFinal

      onResult(transcript, confidence, isFinal)
    }

    this.recognition.onerror = (event: any) => {
      onError(event.error)
    }

    this.recognition.onend = () => {
      this.isListening = false
    }

    this.recognition.start()
    this.isListening = true
  }

  stopListening() {
    if (this.recognition && this.isListening) {
      this.recognition.stop()
      this.isListening = false
    }
  }

  speak(
    text: string,
    options: {
      rate?: number
      pitch?: number
      volume?: number
      voice?: string
    } = {}
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.synthesis) {
        reject(new Error('Speech synthesis not available'))
        return
      }

      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = options.rate || 1.0
      utterance.pitch = options.pitch || 1.0
      utterance.volume = options.volume || 0.8

      // Select appropriate voice
      const voices = this.synthesis.getVoices()
      if (options.voice) {
        const selectedVoice = voices.find(voice => voice.name.includes(options.voice!))
        if (selectedVoice) utterance.voice = selectedVoice
      } else {
        // Select a pleasant default voice
        const preferredVoice = voices.find(voice => 
          voice.lang.startsWith('en') && 
          (voice.name.includes('Female') || voice.name.includes('Google'))
        )
        if (preferredVoice) utterance.voice = preferredVoice
      }

      utterance.onend = () => resolve()
      utterance.onerror = (event) => reject(event.error)

      this.synthesis.speak(utterance)
    })
  }

  getAvailableVoices(): SpeechSynthesisVoice[] {
    return this.synthesis?.getVoices() || []
  }

  isSupported(): boolean {
    return !!(this.recognition && this.synthesis)
  }
}
