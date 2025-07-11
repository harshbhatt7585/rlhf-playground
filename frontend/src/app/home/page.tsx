'use client';

import React, { useState } from 'react';

export default function PromptPage() {
  const [prompt, setPrompt] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    // TODO: handle agent prompt submission
    console.log('Submitted prompt:', prompt);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-black">
      <form onSubmit={handleSubmit} className="w-full max-w-md">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt..."
          className="w-full px-6 py-4 bg-gray-800 text-white placeholder-gray-500 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
        />
        <button
          type="submit"
          className="mt-4 w-full px-6 py-4 bg-gradient-to-r from-blue-500 to-green-400 text-white font-semibold rounded-lg hover:from-blue-600 hover:to-green-500 transform hover:-translate-y-1 hover:shadow-lg transition"
        >
          Submit
        </button>
      </form>
    </div>
  );
}
