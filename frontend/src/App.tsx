import React, { useState, useEffect, useCallback } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from 'react-query'
import { Toaster } from 'react-hot-toast'
import { motion, AnimatePresence } from 'framer-motion'

// Components
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Chat from './pages/Chat'
import Settings from './pages/Settings'
import Analytics from './pages/Analytics'
import Models from './pages/Models'
import VectorDB from './pages/VectorDB'
import Logs from './pages/Logs'
import SystemStatus from './pages/SystemStatus'

// Hooks
import { useWebSocket } from './hooks/useWebSocket'
import { useTheme } from './hooks/useTheme'
import { useSystemStatus } from './hooks/useSystemStatus'

// Context
import { SystemProvider } from './context/SystemContext'
import { ChatProvider } from './context/ChatContext'

// Styles
import './App.css'

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
})

const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const { theme, toggleTheme } = useTheme()
  const { status, health } = useSystemStatus()
  const { isConnected, sendMessage, lastMessage } = useWebSocket('ws://localhost:8000/ws')

  // Initialize app
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Check system health
        const response = await fetch('/api/health')
        if (response.ok) {
          console.log('🌊 DeepBlue 2.0 system is healthy')
        }
        setIsLoading(false)
      } catch (error) {
        console.error('Failed to initialize app:', error)
        setIsLoading(false)
      }
    }

    initializeApp()
  }, [])

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      console.log('WebSocket message received:', lastMessage)
    }
  }, [lastMessage])

  const toggleSidebar = useCallback(() => {
    setSidebarOpen(prev => !prev)
  }, [])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <div className="text-6xl mb-4">🌊</div>
          <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-2">
            DeepBlue 2.0
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mb-4">
            The Ultimate AI System
          </p>
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
        </motion.div>
      </div>
    )
  }

  return (
    <QueryClientProvider client={queryClient}>
      <SystemProvider>
        <ChatProvider>
          <Router>
            <div className={`min-h-screen ${theme === 'dark' ? 'dark' : ''}`}>
              <div className="bg-white dark:bg-gray-900 text-gray-900 dark:text-white transition-colors duration-300">
                {/* Header */}
                <Header
                  sidebarOpen={sidebarOpen}
                  toggleSidebar={toggleSidebar}
                  theme={theme}
                  toggleTheme={toggleTheme}
                  systemStatus={status}
                  isConnected={isConnected}
                />

                {/* Sidebar */}
                <Sidebar
                  open={sidebarOpen}
                  onClose={() => setSidebarOpen(false)}
                />

                {/* Main Content */}
                <main className={`transition-all duration-300 ${sidebarOpen ? 'ml-64' : 'ml-0'}`}>
                  <AnimatePresence mode="wait">
                    <Routes>
                      <Route path="/" element={<Navigate to="/dashboard" replace />} />
                      <Route path="/dashboard" element={<Dashboard />} />
                      <Route path="/chat" element={<Chat />} />
                      <Route path="/settings" element={<Settings />} />
                      <Route path="/analytics" element={<Analytics />} />
                      <Route path="/models" element={<Models />} />
                      <Route path="/vectordb" element={<VectorDB />} />
                      <Route path="/logs" element={<Logs />} />
                      <Route path="/status" element={<SystemStatus />} />
                    </Routes>
                  </AnimatePresence>
                </main>

                {/* Toast Notifications */}
                <Toaster
                  position="top-right"
                  toastOptions={{
                    duration: 4000,
                    style: {
                      background: theme === 'dark' ? '#374151' : '#ffffff',
                      color: theme === 'dark' ? '#ffffff' : '#000000',
                    },
                  }}
                />
              </div>
            </div>
          </Router>
        </ChatProvider>
      </SystemProvider>
    </QueryClientProvider>
  )
}

export default App

