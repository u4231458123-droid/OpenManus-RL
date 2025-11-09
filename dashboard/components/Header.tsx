'use client'

import Link from 'next/link'
import { Bot, Activity, Settings } from 'lucide-react'

export default function Header() {
  return (
    <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">NeXifyAI</h1>
              <p className="text-xs text-gray-400">RL-Powered Assistant</p>
            </div>
          </Link>

          {/* Navigation */}
          <nav className="flex items-center gap-6">
            <Link
              href="/analytics"
              className="flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
            >
              <Activity className="w-5 h-5" />
              <span>Analytics</span>
            </Link>
            <Link
              href="/settings"
              className="flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
            >
              <Settings className="w-5 h-5" />
              <span>Settings</span>
            </Link>
          </nav>
        </div>
      </div>
    </header>
  )
}
