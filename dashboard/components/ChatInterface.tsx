'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, Bot, User, AlertCircle } from 'lucide-react'
import { useAuthStore } from '@/lib/auth-store'
import { AIManager } from '@/lib/ai/ai-manager'
import toast from 'react-hot-toast'
import Link from 'next/link'
import ReactMarkdown from 'react-markdown'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  thinking?: string
  tools_used?: string[]
}

export default function ChatInterface() {
  const { apiKeys } = useAuthStore()
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Hallo! Ich bin NeXifyAI, Ihr intelligenter KI-Assistent. Wie kann ich Ihnen heute helfen?',
      timestamp: new Date()
    }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingContent])

  const hasApiKeys = !!(apiKeys.openai || apiKeys.anthropic)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    if (!hasApiKeys) {
      toast.error('Bitte fügen Sie zuerst Ihre API-Schlüssel in den Einstellungen hinzu')
      return
    }

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setIsStreaming(true)
    setStreamingContent('')

    try {
      const aiManager = new AIManager(apiKeys.openai, apiKeys.anthropic)

      // System prompt with tool awareness
      const systemPrompt = `Du bist NeXifyAI, ein intelligenter KI-Assistent mit Zugriff auf verschiedene Tools.
Du kannst:
- Web-Suche durchführen (google_search)
- Wikipedia durchsuchen (wikipedia_knowledge_searcher)
- Arxiv Papers finden (arxiv_paper_searcher)
- Bilder analysieren (image_captioner, advanced_object_detector)
- Python-Code generieren (python_code_generator)
- URLs extrahieren (url_text_extractor)

Antworte präzise, hilfreich und in deutscher Sprache.`

      const chatMessages = [
        { role: 'system' as const, content: systemPrompt },
        ...messages.map(m => ({ role: m.role, content: m.content })),
        { role: 'user' as const, content: input }
      ]

      let fullResponse = ''

      await aiManager.streamChat(
        chatMessages,
        (chunk) => {
          fullResponse += chunk
          setStreamingContent(fullResponse)
        }
      )

      const assistantMessage: Message = {
        role: 'assistant',
        content: fullResponse || 'Entschuldigung, ich konnte keine Antwort generieren.',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
      setStreamingContent('')
    } catch (error: any) {
      console.error('Chat error:', error)
      toast.error(error.message || 'Fehler bei der AI-Anfrage')
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Fehler: ${error.message}\n\nBitte überprüfen Sie Ihre API-Schlüssel in den Einstellungen.`,
        timestamp: new Date()
      }])
    } finally {
      setIsLoading(false)
      setIsStreaming(false)
      setStreamingContent('')
    }
  }

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-2xl border border-gray-700 overflow-hidden">
      {!hasApiKeys && (
        <div className="bg-yellow-900/30 border-b border-yellow-700 p-4">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-400" />
            <p className="text-yellow-300 text-sm">
              Sie haben noch keine API-Schlüssel konfiguriert.{' '}
              <Link href="/settings" className="underline font-semibold hover:text-yellow-200">
                Jetzt in Einstellungen hinzufügen
              </Link>
            </p>
          </div>
        </div>
      )}

      {/* Messages Container */}
      <div className="h-[600px] overflow-y-auto p-6 space-y-6">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex items-start gap-4 ${
              message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
            }`}
          >
            {/* Avatar */}
            <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
              message.role === 'user'
                ? 'bg-blue-600'
                : 'animated-gradient'
            }`}>
              {message.role === 'user' ? (
                <User className="w-5 h-5 text-white" />
              ) : (
                <Bot className="w-5 h-5 text-white" />
              )}
            </div>

            {/* Message Content */}
            <div className={`flex-1 ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
              <div className={`inline-block p-4 rounded-2xl max-w-[80%] ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-100'
              }`}>
                {message.thinking && (
                  <div className="mb-2 text-xs text-purple-400 italic border-l-2 border-purple-500 pl-2">
                    💭 {message.thinking}
                  </div>
                )}
                <div className="prose prose-invert prose-sm max-w-none">
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>
                {message.tools_used && message.tools_used.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {message.tools_used.map((tool, i) => (
                      <span key={i} className="text-xs bg-purple-900/50 px-2 py-1 rounded border border-purple-700">
                        🔧 {tool}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {message.timestamp.toLocaleTimeString('de-DE')}
              </div>
            </div>
          </div>
        ))}

        {isStreaming && streamingContent && (
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 w-10 h-10 rounded-full animated-gradient flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1">
              <div className="inline-block p-4 rounded-2xl bg-gray-700 max-w-[80%]">
                <div className="prose prose-invert prose-sm max-w-none">
                  <ReactMarkdown>{streamingContent}</ReactMarkdown>
                </div>
                <Loader2 className="w-4 h-4 text-purple-400 animate-spin mt-2" />
              </div>
            </div>
          </div>
        )}

        {isLoading && !isStreaming && (
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 w-10 h-10 rounded-full animated-gradient flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1">
              <div className="inline-block p-4 rounded-2xl bg-gray-700">
                <Loader2 className="w-5 h-5 text-purple-400 animate-spin" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="p-6 bg-gray-900/50 border-t border-gray-700">
        <div className="flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Stelle mir eine Frage..."
            disabled={isLoading}
            className="flex-1 bg-gray-800 text-white placeholder-gray-400 px-6 py-4 rounded-xl border border-gray-700 focus:border-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 disabled:opacity-50 transition-all"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-8 py-4 rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-105 flex items-center gap-2"
          >
            <Send className="w-5 h-5" />
            Senden
          </button>
        </div>
      </form>
    </div>
  )
}
