import { Suspense } from 'react'
import ChatInterface from '@/components/ChatInterface'
import Header from '@/components/Header'

export const dynamic = 'force-dynamic'
export const revalidate = 0

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <Header />

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <h1 className="text-6xl font-bold gradient-text mb-4 glow">
              NeXifyAI
            </h1>
            <p className="text-xl text-gray-300 mb-2">
              Intelligente KI-Assistenz mit Tool-Integration
            </p>
            <p className="text-gray-400">
              Powered by Reinforcement Learning & Advanced Reasoning
            </p>
          </div>

          {/* Main Chat Interface */}
          <Suspense fallback={<LoadingChat />}>
            <ChatInterface />
          </Suspense>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
            <FeatureCard
              icon="🔍"
              title="Web Search"
              description="Durchsuche das Internet in Echtzeit"
            />
            <FeatureCard
              icon="📚"
              title="Knowledge Base"
              description="Zugriff auf Wikipedia & Arxiv"
            />
            <FeatureCard
              icon="🛠️"
              title="Tool Integration"
              description="Python Code, Bildanalyse & mehr"
            />
            <FeatureCard
              icon="🧠"
              title="Reasoning"
              description="Multi-Turn Reasoning & Planning"
            />
            <FeatureCard
              icon="📊"
              title="Analytics"
              description="Training Metriken & Performance"
            />
            <FeatureCard
              icon="🔄"
              title="Continuous Learning"
              description="RL-basierte Verbesserung"
            />
          </div>
        </div>
      </div>
    </main>
  )
}

function LoadingChat() {
  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-2xl p-8 border border-gray-700">
      <div className="animate-pulse space-y-4">
        <div className="h-12 bg-gray-700 rounded"></div>
        <div className="h-64 bg-gray-700 rounded"></div>
        <div className="h-12 bg-gray-700 rounded"></div>
      </div>
    </div>
  )
}

interface FeatureCardProps {
  icon: string
  title: string
  description: string
}

function FeatureCard({ icon, title, description }: FeatureCardProps) {
  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-blue-500 transition-all hover:scale-105">
      <div className="text-4xl mb-3">{icon}</div>
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <p className="text-gray-400 text-sm">{description}</p>
    </div>
  )
}
