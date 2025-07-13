'use client';

import React, { useState, FormEvent } from 'react';

interface PreferenceExample {
  prompt: string;
  chosen: string;
  rejected: string;
}

interface PreferenceGenRes {
  generated: PreferenceExample[];
}

export default function Generate() {
  const [promptText, setPromptText] = useState('');
  const [chosen, setChosen] = useState('');
  const [rejected, setRejected] = useState('');
  const [examples, setExamples] = useState<PreferenceExample[]>([]);
  const [generated, setGenerated] = useState<PreferenceExample[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const addExample = () => {
    if (!promptText.trim() || !chosen.trim() || !rejected.trim()) {
      setError('Please fill out all fields before adding');
      return;
    }
    setExamples([...examples, { prompt: promptText, chosen, rejected }]);
    setPromptText('');
    setChosen('');
    setRejected('');
    setError('');
  };

  const clearExamples = () => {
    setExamples([]);
  };

  const deleteExample = (index: number) => {
    setExamples(examples.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (examples.length === 0) {
      setError('Please add at least one example');
      return;
    }
    setIsLoading(true);
    setError('');

    try {
      const body = { seed_examples: examples, num_generations: 3 };
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/generate/preferences`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        }
      );
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data: PreferenceGenRes = await res.json();
      setGenerated(data.generated);
    } catch (err: any) {
      console.error(err);
      setError('Failed to generate preferences. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black flex flex-col items-center py-12 px-6">
      <div className="w-full max-w-2xl">
        <div className="mb-10 text-center">
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-green-300">
            Create Example Dataset
          </h1>
          <p className="mt-4 text-gray-400">
            Add seed examples below and generate new preference pairs.
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="bg-gray-800 p-6 rounded-2xl shadow-lg space-y-6"
          aria-label="Model alignment prompt form"
        >
          <div className="space-y-4">
            <textarea
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              placeholder="Enter your prompt..."
              className="w-full px-5 py-3 bg-gray-700 text-white placeholder-gray-400 rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200 resize-y min-h-[100px]"
              disabled={isLoading}
            />
            <textarea
              value={chosen}
              onChange={(e) => setChosen(e.target.value)}
              placeholder="Chosen (Gen Z style)..."
              className="w-full px-5 py-3 bg-gray-700 text-white placeholder-gray-400 rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-green-500 transition duration-200 resize-y min-h-[80px]"
              disabled={isLoading}
            />
            <textarea
              value={rejected}
              onChange={(e) => setRejected(e.target.value)}
              placeholder="Rejected (Formal style)..."
              className="w-full px-5 py-3 bg-gray-700 text-white placeholder-gray-400 rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-red-500 transition duration-200 resize-y min-h-[80px]"
              disabled={isLoading}
            />
            {error && (
              <p className="text-center text-sm text-red-400" role="alert">
                {error}
              </p>
            )}
          </div>

          <div className="flex flex-wrap gap-4 justify-between">
            <button
              type="button"
              onClick={addExample}
              disabled={isLoading}
              className="flex-1 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-lg shadow hover:shadow-md transition"
            >
              Add Example
            </button>
            <button
              type="button"
              onClick={clearExamples}
              disabled={isLoading}
              className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white font-semibold rounded-lg shadow hover:shadow-md transition"
            >
              Clear All
            </button>
            <button
              type="submit"
              disabled={isLoading || examples.length === 0}
              className="flex-1 px-4 py-2 bg-gradient-to-r from-blue-500 to-green-400 text-white font-semibold rounded-lg shadow-lg transform transition hover:-translate-y-1 hover:shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Generating...' : 'Generate'}
            </button>
          </div>
        </form>

        {examples.length > 0 && (
          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
            {examples.map((ex, idx) => (
              <div
                key={idx}
                className="relative p-4 bg-gray-800 rounded-xl border border-gray-700 shadow-inner hover:shadow-lg transition"
              >
                <button
                  onClick={() => deleteExample(idx)}
                  className="absolute top-3 right-3 text-gray-400 hover:text-red-400"
                  aria-label="Delete example"
                >
                  ×
                </button>
                <p className="text-gray-200 mb-2">
                  <span className="font-medium">Prompt:</span> {ex.prompt}
                </p>
                <p className="text-green-300 mb-1">
                  <span className="font-medium">Chosen:</span> {ex.chosen}
                </p>
                <p className="text-red-300">
                  <span className="font-medium">Rejected:</span> {ex.rejected}
                </p>
              </div>
            ))}
          </div>
        )}

        {generated.length > 0 && (
          <div className="mt-8">
            <h2 className="text-2xl font-semibold text-white mb-4">
              Generated Preferences
            </h2>
            <div className="space-y-4">
              {generated.map((item, idx) => (
                <div
                  key={idx}
                  className="p-4 bg-gray-800 rounded-xl border border-gray-700 shadow-inner"
                >
                  <p className="text-gray-200 mb-1">
                    <span className="font-medium">Prompt:</span> {item.prompt}
                  </p>
                  <p className="text-green-300 mb-1">
                    <span className="font-medium">Chosen:</span> {item.chosen}
                  </p>
                  <p className="text-red-300">
                    <span className="font-medium">Rejected:</span> {item.rejected}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}