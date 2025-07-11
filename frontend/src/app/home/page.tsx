'use client';

import React, { useState } from 'react';

export default function PromptPage() {
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    setIsLoading(true);
    try {
      // Simulate API call or agent prompt submission
      console.log('Submitted prompt:', prompt);
      // Reset form after successful submission
      setPrompt('');
    } catch (err) {
      setError('Failed to submit prompt. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black flex flex-col items-center justify-center px-4 py-12">
      <h1 className="text-center text-3xl sm:text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-green-300 mb-8">
        Align Your Model
      </h1>
      <p className="text-center text-gray-300 text-base sm:text-lg mb-10 max-w-2xl">
        Describe how you want to align your model to get tailored responses that match your goals.
      </p>
      
      <form 
        onSubmit={handleSubmit} 
        className="w-full max-w-lg space-y-4"
        aria-label="Model alignment prompt form"
      >
        <div className="relative">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt..."
            className="w-full px-6 py-4 bg-gray-800 text-white placeholder-gray-400 rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300 ease-in-out"
            aria-label="Prompt input"
            disabled={isLoading}
          />
          {error && (
            <p className="mt-2 text-sm text-red-400" role="alert">
              {error}
            </p>
          )}
        </div>
        
        <button
          type="submit"
          disabled={isLoading}
          className={`w-full px-6 py-4 bg-gradient-to-r from-blue-500 to-green-400 text-white font-semibold rounded-lg transition-all duration-300 ease-in-out transform hover:-translate-y-1 hover:shadow-xl ${
            isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:from-blue-600 hover:to-green-500'
          }`}
          aria-label="Submit prompt"
        >
          {isLoading ? (
            <span className="flex items-center justify-center">
              <svg
                className="animate-spin h-5 w-5 mr-2 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v8h8a8 8 0 01-16 0z"
                />
              </svg>
              Submitting...
            </span>
          ) : (
            'Submit Prompt'
          )}
        </button>
      </form>
    </div>
  );
}