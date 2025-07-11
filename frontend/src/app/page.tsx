'use client';

import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface StatItemProps {
  number: string;
  label: string;
}

interface FeatureCardProps {
  icon: string;
  title: string;
  description: string;
}

const StatItem: React.FC<StatItemProps> = ({ number, label }) => (
  <div className="flex flex-col items-center">
    <div className="text-5xl font-extrabold text-cyan-400 mb-2">{number}</div>
    <div className="text-lg text-gray-400">{label}</div>
  </div>
);

const FeatureCard: React.FC<FeatureCardProps> = ({ icon, title, description }) => (
  <div className="relative bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-10 transition-all duration-300 hover:-translate-y-3 hover:shadow-2xl hover:shadow-cyan-500/20 overflow-hidden group">
    <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500"></div>
    <div className="w-15 h-15 bg-gradient-to-br from-cyan-400 to-purple-500 rounded-2xl flex items-center justify-center mb-6 text-2xl">
      {icon}
    </div>
    <h3 className="text-2xl font-semibold text-white mb-4">{title}</h3>
    <p className="text-gray-400 leading-relaxed">{description}</p>
  </div>
);

const NeuralNetworkVisualization: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const animationIdRef = useRef<number | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const scene = new THREE.Scene();
    sceneRef.current = scene;
    
    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    rendererRef.current = renderer;
    
    renderer.setSize(400, 400);
    renderer.setClearColor(0x000000, 0);
    containerRef.current.appendChild(renderer.domElement);

    // Create nodes
    const nodes: Array<{ mesh: THREE.Mesh; layer: number; index: number }> = [];
    const connections: THREE.Line[] = [];
    
    // Node layers
    const layers = [4, 6, 6, 4, 1];
    let nodeIndex = 0;
    
    layers.forEach((layerSize, layerIndex) => {
      for (let i = 0; i < layerSize; i++) {
        const geometry = new THREE.SphereGeometry(0.05, 16, 16);
        const material = new THREE.MeshBasicMaterial({ 
          color: new THREE.Color().setHSL(0.6 + Math.random() * 0.4, 0.8, 0.6),
          transparent: true,
          opacity: 0.8
        });
        const node = new THREE.Mesh(geometry, material);
        
        const x = (layerIndex - layers.length / 2) * 0.8;
        const y = (i - layerSize / 2) * 0.4;
        const z = (Math.random() - 0.5) * 0.3;
        
        node.position.set(x, y, z);
        scene.add(node);
        nodes.push({ mesh: node, layer: layerIndex, index: nodeIndex++ });
      }
    });

    // Create connections
    nodes.forEach(node => {
      if (node.layer < layers.length - 1) {
        const nextLayerNodes = nodes.filter(n => n.layer === node.layer + 1);
        nextLayerNodes.forEach(nextNode => {
          if (Math.random() > 0.3) {
            const geometry = new THREE.BufferGeometry().setFromPoints([
              node.mesh.position,
              nextNode.mesh.position
            ]);
            const material = new THREE.LineBasicMaterial({ 
              color: 0x00d4ff,
              transparent: true,
              opacity: 0.3
            });
            const line = new THREE.Line(geometry, material);
            scene.add(line);
            connections.push(line);
          }
        });
      }
    });

    camera.position.z = 3;

    // Animation
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      
      // Rotate the entire network
      scene.rotation.y += 0.005;
      scene.rotation.x += 0.002;
      
      // Animate nodes
      nodes.forEach((node, index) => {
        const time = Date.now() * 0.001;
        (node.mesh.material as THREE.MeshBasicMaterial).opacity = 0.5 + Math.sin(time + index * 0.1) * 0.3;
      });
      
      // Animate connections
      connections.forEach((connection, index) => {
        const time = Date.now() * 0.001;
        (connection.material as THREE.LineBasicMaterial).opacity = 0.1 + Math.sin(time + index * 0.05) * 0.2;
      });
      
      renderer.render(scene, camera);
    };
    
    animate();

    // Cleanup
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return <div ref={containerRef} className="w-96 h-96 flex items-center justify-center" />;
};

const Navigation: React.FC = () => {
  const handleScroll = (e: React.MouseEvent<HTMLAnchorElement>, targetId: string) => {
    e.preventDefault();
    const target = document.querySelector(targetId);
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    const handleScrollEffect = () => {
      const nav = document.querySelector('nav');
      if (nav) {
        if (window.scrollY > 100) {
          nav.style.background = 'rgba(10, 10, 10, 0.95)';
        } else {
          nav.style.background = 'rgba(10, 10, 10, 0.7)';
        }
      }
    };

    window.addEventListener('scroll', handleScrollEffect);
    return () => window.removeEventListener('scroll', handleScrollEffect);
  }, []);

  return (
    <nav className="fixed top-0 w-full backdrop-blur-xl z-50 py-4 transition-all duration-300" style={{ background: 'rgba(10, 10, 10, 0.7)' }}>
      <div className="max-w-7xl mx-auto px-8">
        <div className="flex justify-between items-center">
          <div className="text-3xl font-bold bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            AlignAI
          </div>
          <ul className="hidden md:flex space-x-8">
            <li><a href="#home" onClick={(e) => handleScroll(e, '#home')} className="text-white hover:text-cyan-400 transition-all duration-300 font-medium hover:-translate-y-1">Home</a></li>
            <li><a href="#features" onClick={(e) => handleScroll(e, '#features')} className="text-white hover:text-cyan-400 transition-all duration-300 font-medium hover:-translate-y-1">Features</a></li>
            <li><a href="#about" onClick={(e) => handleScroll(e, '#about')} className="text-white hover:text-cyan-400 transition-all duration-300 font-medium hover:-translate-y-1">About</a></li>
            <li><a href="#contact" onClick={(e) => handleScroll(e, '#contact')} className="text-white hover:text-cyan-400 transition-all duration-300 font-medium hover:-translate-y-1">Contact</a></li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default function Home() {
  const features: FeatureCardProps[] = [
    {
      icon: '🧠',
      title: 'Advanced NLP Processing',
      description: 'Leverage state-of-the-art natural language processing to understand and align model behavior with human intentions and values.'
    },
    {
      icon: '⚡',
      title: 'Real-time Alignment',
      description: 'Monitor and adjust model alignment in real-time, ensuring consistent performance and safety across all interactions.'
    },
    {
      icon: '🔒',
      title: 'Safety First',
      description: 'Built-in safety mechanisms prevent harmful outputs and ensure your AI systems remain trustworthy and reliable.'
    },
    {
      icon: '📊',
      title: 'Comprehensive Analytics',
      description: 'Detailed insights and metrics help you understand model performance and track alignment improvements over time.'
    },
    {
      icon: '🔧',
      title: 'Easy Integration',
      description: 'Seamlessly integrate with your existing AI infrastructure through our robust APIs and developer-friendly tools.'
    },
    {
      icon: '🌍',
      title: 'Scalable Solution',
      description: 'From small experiments to enterprise-scale deployments, our platform grows with your needs.'
    }
  ];

  const stats: StatItemProps[] = [
    { number: '99.9%', label: 'Alignment Accuracy' },
    { number: '10x', label: 'Faster Processing' },
    { number: '500+', label: 'Models Aligned' },
    { number: '24/7', label: 'Continuous Monitoring' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 text-white overflow-x-hidden">
      <Navigation />

      {/* Hero Section */}
      <section className="min-h-screen flex items-center relative overflow-hidden" id="home">
        <div className="max-w-7xl mx-auto px-8 w-full">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div className="space-y-8 animate-fade-in-up">
              <h1 className="text-4xl md:text-6xl lg:text-7xl font-extrabold leading-tight bg-gradient-to-r from-white via-cyan-400 to-purple-500 bg-clip-text text-transparent">
                Intelligent Model Alignment Through NLP
              </h1>
              <p className="text-xl md:text-2xl text-gray-300 leading-relaxed">
                Transform your AI models with our cutting-edge NLP alignment technology. 
                Achieve unprecedented accuracy, safety, and reliability in your AI systems.
              </p>
              <div className="flex flex-col sm:flex-row gap-6">
                <button className="px-8 py-4 bg-gradient-to-r from-cyan-400 to-purple-500 text-white font-semibold rounded-full text-lg transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl hover:shadow-cyan-500/30 flex items-center gap-2">
                  🚀 Get Started
                </button>
                <button className="px-8 py-4 bg-transparent border-2 border-white text-white font-semibold rounded-full text-lg transition-all duration-300 hover:bg-white hover:text-gray-900 hover:-translate-y-1 flex items-center gap-2">
                  📖 Learn More
                </button>
              </div>
            </div>
            <div className="flex justify-center items-center animate-float">
              <NeuralNetworkVisualization />
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-32 bg-white/5 backdrop-blur-sm" id="features">
        <div className="max-w-7xl mx-auto px-8">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-16 bg-gradient-to-r from-white to-cyan-400 bg-clip-text text-transparent">
            Powerful Features
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <FeatureCard key={index} {...feature} />
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-24 bg-gradient-to-r from-cyan-400/10 to-purple-500/10">
        <div className="max-w-7xl mx-auto px-8">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-12 text-center">
            {stats.map((stat, index) => (
              <StatItem key={index} {...stat} />
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 bg-black/50">
        <div className="max-w-7xl mx-auto px-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-8">
            <div className="text-3xl font-bold bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              AlignAI
            </div>
            <ul className="flex flex-wrap gap-8">
              <li><a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors duration-300">Privacy Policy</a></li>
              <li><a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors duration-300">Terms of Service</a></li>
              <li><a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors duration-300">Documentation</a></li>
              <li><a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors duration-300">Support</a></li>
            </ul>
            <div className="text-gray-400">
              <p>&copy; 2025 AlignAI. All rights reserved.</p>
            </div>
          </div>
        </div>
      </footer>

      <style jsx>{`
        @keyframes fade-in-up {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
        }

        .animate-fade-in-up {
          animation: fade-in-up 1s ease-out;
        }

        .animate-float {
          animation: float 6s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}