import React, { useState, useEffect, useRef, Suspense } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Box, Sphere, Torus, Environment, Stars, Sky } from '@react-three/drei'
import { motion, AnimatePresence, useAnimation } from 'framer-motion'
import { useSpring, animated, useTrail } from '@react-spring/web'
import { useGesture } from '@use-gesture/react'
import * as THREE from 'three'

// 3D Components
const FloatingParticles: React.FC = () => {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const { scene } = useThree()
  
  useEffect(() => {
    if (meshRef.current) {
      const count = 1000
      const positions = new Float32Array(count * 3)
      const colors = new Float32Array(count * 3)
      
      for (let i = 0; i < count; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 100
        positions[i * 3 + 1] = (Math.random() - 0.5) * 100
        positions[i * 3 + 2] = (Math.random() - 0.5) * 100
        
        colors[i * 3] = Math.random()
        colors[i * 3 + 1] = Math.random()
        colors[i * 3 + 2] = Math.random()
      }
      
      meshRef.current.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      meshRef.current.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    }
  }, [])
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = state.clock.elapsedTime * 0.1
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.05
    }
  })
  
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, 1000]}>
      <sphereGeometry args={[0.1, 8, 8]} />
      <meshBasicMaterial vertexColors />
    </instancedMesh>
  )
}

const NeuralNetwork: React.FC = () => {
  const [nodes, setNodes] = useState<THREE.Vector3[]>([])
  const [connections, setConnections] = useState<[THREE.Vector3, THREE.Vector3][]>([])
  
  useEffect(() => {
    const nodeCount = 50
    const newNodes: THREE.Vector3[] = []
    const newConnections: [THREE.Vector3, THREE.Vector3][] = []
    
    // Generate nodes
    for (let i = 0; i < nodeCount; i++) {
      newNodes.push(new THREE.Vector3(
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20
      ))
    }
    
    // Generate connections
    for (let i = 0; i < nodeCount; i++) {
      for (let j = i + 1; j < nodeCount; j++) {
        if (Math.random() < 0.1) {
          newConnections.push([newNodes[i], newNodes[j]])
        }
      }
    }
    
    setNodes(newNodes)
    setConnections(newConnections)
  }, [])
  
  return (
    <group>
      {nodes.map((node, index) => (
        <Sphere key={index} position={node} args={[0.2, 8, 8]}>
          <meshBasicMaterial color="#3b82f6" />
        </Sphere>
      ))}
      {connections.map(([start, end], index) => (
        <group key={index}>
          <Box position={[(start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2]}>
            <meshBasicMaterial color="#60a5fa" />
          </Box>
        </group>
      ))}
    </group>
  )
}

const HolographicDisplay: React.FC<{ text: string }> = ({ text }) => {
  const meshRef = useRef<THREE.Mesh>(null)
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.5
      meshRef.current.material.opacity = 0.8 + Math.sin(state.clock.elapsedTime * 2) * 0.2
    }
  })
  
  return (
    <group>
      <Text
        ref={meshRef}
        position={[0, 0, 0]}
        fontSize={2}
        color="#00ffff"
        anchorX="center"
        anchorY="middle"
      >
        {text}
      </Text>
      <Torus args={[3, 0.1, 16, 100]} position={[0, 0, 0]}>
        <meshBasicMaterial color="#00ffff" transparent opacity={0.3} />
      </Torus>
    </group>
  )
}

// Advanced UI Components
const ParticleBackground: React.FC = () => {
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; vx: number; vy: number }>>([])
  
  useEffect(() => {
    const newParticles = Array.from({ length: 100 }, (_, i) => ({
      id: i,
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 2,
      vy: (Math.random() - 0.5) * 2,
    }))
    setParticles(newParticles)
  }, [])
  
  useEffect(() => {
    const animate = () => {
      setParticles(prev => prev.map(particle => ({
        ...particle,
        x: particle.x + particle.vx,
        y: particle.y + particle.vy,
        vx: particle.x > window.innerWidth || particle.x < 0 ? -particle.vx : particle.vx,
        vy: particle.y > window.innerHeight || particle.y < 0 ? -particle.vy : particle.vy,
      })))
    }
    
    const interval = setInterval(animate, 16)
    return () => clearInterval(interval)
  }, [])
  
  return (
    <div className="fixed inset-0 pointer-events-none">
      {particles.map(particle => (
        <motion.div
          key={particle.id}
          className="absolute w-1 h-1 bg-blue-400 rounded-full"
          style={{ left: particle.x, top: particle.y }}
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.3, 0.8, 0.3],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            delay: particle.id * 0.01,
          }}
        />
      ))}
    </div>
  )
}

const GlitchText: React.FC<{ children: React.ReactNode; className?: string }> = ({ 
  children, 
  className = "" 
}) => {
  const [isGlitching, setIsGlitching] = useState(false)
  
  useEffect(() => {
    const interval = setInterval(() => {
      setIsGlitching(true)
      setTimeout(() => setIsGlitching(false), 100)
    }, 3000)
    
    return () => clearInterval(interval)
  }, [])
  
  return (
    <motion.div
      className={`relative ${className}`}
      animate={isGlitching ? {
        x: [0, -2, 2, -1, 1, 0],
        y: [0, 1, -1, 0.5, -0.5, 0],
      } : {}}
      transition={{ duration: 0.1 }}
    >
      {children}
      {isGlitching && (
        <motion.div
          className="absolute inset-0 bg-red-500 mix-blend-difference opacity-20"
          animate={{ opacity: [0, 0.2, 0] }}
          transition={{ duration: 0.1 }}
        />
      )}
    </motion.div>
  )
}

const MorphingShape: React.FC = () => {
  const [shape, setShape] = useState(0)
  const controls = useAnimation()
  
  useEffect(() => {
    const interval = setInterval(() => {
      setShape(prev => (prev + 1) % 4)
    }, 2000)
    
    return () => clearInterval(interval)
  }, [])
  
  useEffect(() => {
    controls.start({
      scale: [1, 1.2, 1],
      rotate: [0, 180, 360],
      transition: { duration: 1, ease: "easeInOut" }
    })
  }, [shape, controls])
  
  const shapes = [
    { borderRadius: "50%", clipPath: "circle(50%)" },
    { borderRadius: "0%", clipPath: "polygon(50% 0%, 0% 100%, 100% 100%)" },
    { borderRadius: "20%", clipPath: "polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%)" },
    { borderRadius: "10%", clipPath: "polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%)" },
  ]
  
  return (
    <motion.div
      className="w-32 h-32 bg-gradient-to-br from-blue-500 to-purple-600"
      style={shapes[shape]}
      animate={controls}
    />
  )
}

const HolographicButton: React.FC<{ 
  children: React.ReactNode; 
  onClick: () => void;
  className?: string;
}> = ({ children, onClick, className = "" }) => {
  const [isHovered, setIsHovered] = useState(false)
  const [isPressed, setIsPressed] = useState(false)
  
  const springProps = useSpring({
    scale: isPressed ? 0.95 : isHovered ? 1.05 : 1,
    boxShadow: isHovered 
      ? "0 0 20px rgba(59, 130, 246, 0.5), 0 0 40px rgba(59, 130, 246, 0.3)"
      : "0 0 10px rgba(59, 130, 246, 0.3)",
    config: { tension: 300, friction: 20 }
  })
  
  return (
    <animated.button
      className={`relative px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg border border-blue-400 ${className}`}
      style={springProps}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onMouseDown={() => setIsPressed(true)}
      onMouseUp={() => setIsPressed(false)}
      onClick={onClick}
    >
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-lg opacity-0"
        animate={{ opacity: isHovered ? 0.3 : 0 }}
        transition={{ duration: 0.2 }}
      />
      <span className="relative z-10">{children}</span>
    </animated.button>
  )
}

const DataVisualization: React.FC<{ data: number[] }> = ({ data }) => {
  const trail = useTrail(data.length, {
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: { opacity: 1, transform: 'translateY(0px)' },
    config: { tension: 300, friction: 20 }
  })
  
  const maxValue = Math.max(...data)
  
  return (
    <div className="flex items-end space-x-1 h-32">
      {trail.map((style, index) => (
        <animated.div
          key={index}
          className="bg-gradient-to-t from-blue-500 to-cyan-400 rounded-t"
          style={{
            ...style,
            width: '20px',
            height: `${(data[index] / maxValue) * 100}%`,
          }}
        />
      ))}
    </div>
  )
}

const AROverlay: React.FC = () => {
  const [isARActive, setIsARActive] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  useEffect(() => {
    if (isARActive && videoRef.current) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          videoRef.current!.srcObject = stream
        })
        .catch(err => console.error('AR camera access failed:', err))
    }
  }, [isARActive])
  
  return (
    <div className="relative">
      {isARActive && (
        <div className="fixed inset-0 z-50 bg-black">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full h-full object-cover"
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
          />
          <motion.div
            className="absolute top-4 right-4 bg-red-500 text-white px-4 py-2 rounded"
            onClick={() => setIsARActive(false)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Exit AR
          </motion.div>
        </div>
      )}
      
      <HolographicButton onClick={() => setIsARActive(true)}>
        Activate AR
      </HolographicButton>
    </div>
  )
}

const AdvancedUI: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0)
  const [is3DMode, setIs3DMode] = useState(false)
  
  const tabs = [
    { name: 'Dashboard', icon: '📊' },
    { name: 'AI Chat', icon: '🤖' },
    { name: 'Analytics', icon: '📈' },
    { name: '3D View', icon: '🎮' },
    { name: 'AR Mode', icon: '🥽' },
  ]
  
  const sampleData = [65, 78, 45, 89, 56, 78, 90, 67, 45, 78, 89, 56]
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 text-white">
      <ParticleBackground />
      
      {/* Header */}
      <motion.header 
        className="relative z-10 p-6"
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8 }}
      >
        <div className="flex items-center justify-between">
          <GlitchText className="text-4xl font-bold">
            🌊 DeepBlue 2.0
          </GlitchText>
          
          <div className="flex items-center space-x-4">
            <HolographicButton onClick={() => setIs3DMode(!is3DMode)}>
              {is3DMode ? '2D Mode' : '3D Mode'}
            </HolographicButton>
            <AROverlay />
          </div>
        </div>
      </motion.header>
      
      {/* Navigation */}
      <motion.nav 
        className="relative z-10 px-6"
        initial={{ x: -100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        <div className="flex space-x-2">
          {tabs.map((tab, index) => (
            <motion.button
              key={tab.name}
              className={`px-4 py-2 rounded-lg transition-all ${
                activeTab === index 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
              onClick={() => setActiveTab(index)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.name}
            </motion.button>
          ))}
        </div>
      </motion.nav>
      
      {/* Main Content */}
      <main className="relative z-10 p-6">
        <AnimatePresence mode="wait">
          {activeTab === 0 && (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
              className="space-y-6"
            >
              <h2 className="text-3xl font-bold">Advanced Dashboard</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <motion.div 
                  className="bg-gray-800 p-6 rounded-lg"
                  whileHover={{ scale: 1.02 }}
                >
                  <h3 className="text-xl font-semibold mb-4">Performance Metrics</h3>
                  <DataVisualization data={sampleData} />
                </motion.div>
                
                <motion.div 
                  className="bg-gray-800 p-6 rounded-lg"
                  whileHover={{ scale: 1.02 }}
                >
                  <h3 className="text-xl font-semibold mb-4">Morphing Shape</h3>
                  <div className="flex justify-center">
                    <MorphingShape />
                  </div>
                </motion.div>
                
                <motion.div 
                  className="bg-gray-800 p-6 rounded-lg"
                  whileHover={{ scale: 1.02 }}
                >
                  <h3 className="text-xl font-semibold mb-4">AI Status</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>GPT-4:</span>
                      <span className="text-green-400">Active</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Claude-3:</span>
                      <span className="text-green-400">Active</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Gemini:</span>
                      <span className="text-green-400">Active</span>
                    </div>
                  </div>
                </motion.div>
              </div>
            </motion.div>
          )}
          
          {activeTab === 3 && is3DMode && (
            <motion.div
              key="3d-view"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
              className="h-96"
            >
              <Canvas camera={{ position: [0, 0, 10] }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />
                <FloatingParticles />
                <NeuralNetwork />
                <HolographicDisplay text="DeepBlue 2.0" />
                <OrbitControls enableZoom={true} enablePan={true} enableRotate={true} />
                <Environment preset="night" />
                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade />
              </Canvas>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}

export default AdvancedUI

