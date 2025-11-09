import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { AuthProvider } from '@/components/providers/AuthProvider'
import { Toaster } from 'react-hot-toast'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'NeXifyAI - Intelligent AI Assistant',
  description: 'RL-powered AI assistant with advanced tool integration and reasoning capabilities',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="de" className="dark">
      <body className={inter.className}>
        <AuthProvider>
          {children}
          <Toaster
            position="top-right"
            toastOptions={{
              className: 'bg-gray-900 text-white border border-purple-500',
              duration: 3000,
            }}
          />
        </AuthProvider>
      </body>
    </html>
  )
}
