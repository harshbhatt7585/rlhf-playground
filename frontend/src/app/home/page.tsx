'use client'

import { useState, useEffect } from 'react'
import { Play, Pause, RotateCcw, CheckCircle, XCircle, Clock, Zap, Database, Bot, Settings, Info } from 'lucide-react'
import { useRouter } from 'next/navigation'

export default function HomePage() {
  const [dataset, setDataset] = useState('')
  const [model, setModel] = useState('')
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [jobs, setJobs] = useState<any[]>([])
  const router = useRouter()

  
  // Advanced configuration
  const [config, setConfig] = useState({
    per_device_train_batch_size: 1,
    gradient_accumulation_steps: 1,
    num_ppo_epochs: 1,
    num_mini_batches: 1,
    response_length: 128,
    total_episodes: 50,
    learning_rate: 1e-5,
    temperature: 0.7
  })

  // Sample datasets and models
  const sampleDatasets = [
    'ccdv/arxiv-summarization',
    'openai/summarize_from_feedback',
    'Anthropic/hh-rlhf',
    'squad_v2'
  ]
  
  const sampleModels = [
    'facebook/opt-125m',
    'microsoft/DialoGPT-small',
    'google/flan-t5-small',
    'distilbert-base-uncased'
  ]

  useEffect(() => {
    // Simulate progress updates
    if (status && status !== 'Completed' && status !== 'Failed') {
      const interval = setInterval(() => {
        setProgress(prev => Math.min(prev + Math.random() * 10, 95))
      }, 1000)
      return () => clearInterval(interval)
    }
  }, [status])

  const handleSubmit = async () => {
    if (!dataset || !model) return
    
    setLoading(true)
    setError(null)
    setJobId(null)
    setStatus(null)
    setProgress(0)

    try {
      const payload = {
        config: {
          model_name: model,
          reward_model_name: model,
          ...config
        },
        hf_dataset: dataset
      }

      const res = await fetch('/api/train/ppo/train-azure-job', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })

      if (!res.ok) {
        const errText = await res.text()
        throw new Error(errText)
      }

      const data = await res.json()
      setJobId(data.job_id)
      pollStatus(data.job_id)
      
      // Add to jobs list
      setJobs(prev => [...prev, {
        id: data.job_id,
        model,
        dataset,
        startTime: new Date(),
        status: 'Starting...'
      }])

      router.push(`/status/${data.job_id}`)
      
    } catch (err: any) {
      setError(err.message || 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }

  const pollStatus = (id: string) => {
    setStatus('Starting...')
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${id}`)
        if (!res.ok) {
          throw new Error(await res.text())
        }
        const data = await res.json()
        setStatus(data.status)
        if (data.completed) {
          clearInterval(interval)
          setProgress(100)
        }
      } catch {
        clearInterval(interval)
        setError('Failed to fetch status')
      }
    }, 3000)
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'Completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'Failed':
        return <XCircle className="w-5 h-5 text-red-500" />
      case 'Running':
        return <Zap className="w-5 h-5 text-blue-500 animate-pulse" />
      default:
        return <Clock className="w-5 h-5 text-yellow-500" />
    }
  }

  const resetForm = () => {
    setDataset('')
    setModel('')
    setJobId(null)
    setStatus(null)
    setError(null)
    setProgress(0)
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-black-900 via-blue-900 to-blue-900 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4 tracking-tight">
            RLHF Playground
          </h1>
          <p className="text-blue-200 text-lg">
            Train language models with reinforcement learning from human feedback
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Form */}
          <div className="lg:col-span-2">
            <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-white/20">
              <div className="flex items-center gap-3 mb-6">
                <Bot className="w-6 h-6 text-blue-400" />
                <h2 className="text-2xl font-bold text-white">Training Configuration</h2>
              </div>

              <div className="space-y-6">
                {/* Dataset Selection */}
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-blue-200 mb-3">
                    <Database className="w-4 h-4" />
                    HuggingFace Dataset
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Enter dataset ID or select from samples..."
                      value={dataset}
                      onChange={(e) => setDataset(e.target.value)}
                      required
                      className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent transition-all"
                    />
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3">
                    {sampleDatasets.map((ds) => (
                      <button
                        key={ds}
                        type="button"
                        onClick={() => setDataset(ds)}
                        className="px-3 py-1 bg-blue-500/20 text-blue-300 rounded-full text-xs hover:bg-blue-500/30 transition-colors"
                      >
                        {ds}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Model Selection */}
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-blue-200 mb-3">
                    <Bot className="w-4 h-4" />
                    Base Model
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Enter model name or select from samples..."
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      required
                      className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent transition-all"
                    />
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3">
                    {sampleModels.map((m) => (
                      <button
                        key={m}
                        type="button"
                        onClick={() => setModel(m)}
                        className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-xs hover:bg-purple-500/30 transition-colors"
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Advanced Settings Toggle */}
                <div className="flex items-center justify-between">
                  <button
                    type="button"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center gap-2 text-blue-300 hover:text-blue-200 transition-colors"
                  >
                    <Settings className="w-4 h-4" />
                    Advanced Settings
                  </button>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={resetForm}
                      className="p-2 text-gray-400 hover:text-white transition-colors"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Advanced Configuration */}
                {showAdvanced && (
                  <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs text-blue-200 mb-2">Batch Size</label>
                        <input
                          type="number"
                          value={config.per_device_train_batch_size}
                          onChange={(e) => setConfig({...config, per_device_train_batch_size: parseInt(e.target.value)})}
                          className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-blue-200 mb-2">Total Episodes</label>
                        <input
                          type="number"
                          value={config.total_episodes}
                          onChange={(e) => setConfig({...config, total_episodes: parseInt(e.target.value)})}
                          className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-blue-200 mb-2">Learning Rate</label>
                        <input
                          type="number"
                          step="0.00001"
                          value={config.learning_rate}
                          onChange={(e) => setConfig({...config, learning_rate: parseFloat(e.target.value)})}
                          className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-blue-200 mb-2">Temperature</label>
                        <input
                          type="number"
                          step="0.1"
                          value={config.temperature}
                          onChange={(e) => setConfig({...config, temperature: parseFloat(e.target.value)})}
                          className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* Submit Button */}
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={loading}
                  className={`w-full py-4 px-6 rounded-xl font-semibold text-white transition-all duration-200 ${
                    loading 
                      ? 'bg-gray-600 cursor-not-allowed' 
                      : 'bg-gradient-to-r from-blue-700 to-blue-600 hover:from-black-600 hover:to-black-700 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
                  }`}
                >
                  <div className="flex items-center justify-center gap-2">
                    {loading ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                        Submitting...
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5" />
                        Start Training
                      </>
                    )}
                  </div>
                </button>

                {error && (
                  <div className="flex items-center gap-2 p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300">
                    <XCircle className="w-5 h-5" />
                    <p className="text-sm">{error}</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Status Panel */}
          <div className="space-y-6">
            {/* Current Job Status */}
            {jobId && (
              <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-6 shadow-2xl border border-white/20">
                <div className="flex items-center gap-3 mb-4">
                  <Zap className="w-6 h-6 text-yellow-400" />
                  <h3 className="text-xl font-bold text-white">Current Job</h3>
                </div>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-blue-200">Job ID:</span>
                    <span className="text-white font-mono text-sm">{jobId}</span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-blue-200">Status:</span>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(status || '')}
                      <span className="text-white">{status}</span>
                    </div>
                  </div>
                  
                  {status && status !== 'Completed' && status !== 'Failed' && (
                    <div>
                      <div className="flex justify-between text-sm text-blue-200 mb-2">
                        <span>Progress</span>
                        <span>{Math.round(progress)}%</span>
                      </div>
                      <div className="w-full bg-white/10 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${progress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Info Panel */}
            <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-6 shadow-2xl border border-white/20">
              <div className="flex items-center gap-3 mb-4">
                <Info className="w-6 h-6 text-blue-400" />
                <h3 className="text-xl font-bold text-white">About RLHF</h3>
              </div>
              
              <div className="space-y-3 text-sm text-blue-200">
                <p>
                  Reinforcement Learning from Human Feedback (RLHF) fine-tunes language models using human preferences.
                </p>
                <p>
                  The process involves training a reward model on human feedback, then using PPO to optimize the language model.
                </p>
                <div className="flex items-center gap-2 mt-4 p-3 bg-blue-500/20 rounded-lg">
                  <Zap className="w-4 h-4 text-blue-400" />
                  <span className="text-blue-300 text-xs">Training typically takes 10-30 minutes for small models</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}