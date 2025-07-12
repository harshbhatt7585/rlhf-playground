'use client'

import { useState } from 'react'

export default function HomePage() {
  const [dataset, setDataset] = useState('')
  const [model, setModel] = useState('')
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setJobId(null)
    setStatus(null)

    try {
      const payload = {
        config: {
          model_name: model,
          reward_model_name: model,
          per_device_train_batch_size: 1,
          gradient_accumulation_steps: 1,
          num_ppo_epochs: 1,
          num_mini_batches: 1,
          response_length: 128,
          total_episodes: 50
        },
        hf_dataset: dataset
      }

      const res = await fetch('/api/train/ppo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const errText = await res.text()
        throw new Error(errText)
      }

      const data = await res.json()
      setJobId(data.job_id)
      pollStatus(data.job_id)
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
        if (data.completed) clearInterval(interval)
      } catch {
        clearInterval(interval)
        setError('Failed to fetch status')
      }
    }, 3000)
  }

  return (
    <main className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-lg p-8 space-y-6">
        <h1 className="text-2xl font-bold text-center">RLHF Playground</h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">
              HF Dataset ID
            </label>
            <input
              type="text"
              placeholder="e.g. ccdv/arxiv-summarization"
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring focus:ring-indigo-200"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Model Name or Link
            </label>
            <input
              type="text"
              placeholder="e.g. facebook/opt-125m"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring focus:ring-indigo-200"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className={`w-full py-2 px-4 rounded-md text-white font-semibold ${
              loading ? 'bg-indigo-300' : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
          >
            {loading ? 'Submitting...' : 'Start Training'}
          </button>

          {error && (
            <p className="text-sm text-red-600 text-center">{error}</p>
          )}
        </form>

        {jobId && (
          <div className="mt-6 bg-gray-100 p-4 rounded-md space-y-2">
            <p>
              <strong>Job ID:</strong> {jobId}
            </p>
            <p>
              <strong>Status:</strong> {status}
            </p>
          </div>
        )}
      </div>
    </main>
  )
}
