'use client';

import { useState } from 'react';
import { ResearchInput } from '@/components/research-input';
import { ResearchResult } from '@/components/research-result';
import { ResearchStatus } from '@/components/research-status';

export default function Home() {
  const [status, setStatus] = useState<'idle' | 'loading' | 'complete' | 'error'>('idle');
  const [report, setReport] = useState<string>('');
  const [error, setError] = useState<string>('');
  
  const handleSubmit = async (query: string) => {
    setStatus('loading');
    setReport('');
    setError('');
    
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

      const response = await fetch(`${backendUrl}/api/research`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Request failed with status ${response.status}`);
      }
      
      const data = await response.json();
      setReport(data.report);
      setStatus('complete');
    } catch (err: any) {
      console.error('Research error:', err);
      setError(err.message || 'An unexpected error occurred');
      setStatus('error');
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-background to-secondary/20 flex flex-col">
      <header className="p-6 flex justify-between items-center max-w-7xl mx-auto w-full">
        <div className="flex items-center gap-2 font-semibold text-lg">
          <div className="h-6 w-6 rounded-full bg-primary" />
          <span>Isara Research</span>
        </div>
      </header>

      <div className="flex-1 flex flex-col items-center justify-center p-4 sm:p-8">
        {status === 'idle' && (
          <ResearchInput onSubmit={handleSubmit} isLoading={false} />
        )}
        
        {status === 'loading' && (
          <ResearchStatus />
        )}
        
        {status === 'error' && (
          <div className="w-full max-w-2xl mx-auto text-center space-y-6">
            <div className="p-8 rounded-xl bg-red-500/10 border border-red-500/20">
              <h3 className="text-xl font-semibold text-red-500 mb-2">Research Failed</h3>
              <p className="text-muted-foreground mb-4">{error}</p>
              <button 
                onClick={() => setStatus('idle')}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity"
              >
                Try Again
              </button>
            </div>
          </div>
        )}
        
        {status === 'complete' && (
          <div className="w-full space-y-6">
            <div className="max-w-4xl mx-auto flex justify-start">
              <button 
                onClick={() => setStatus('idle')}
                className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors"
              >
                ← New Research
              </button>
            </div>
            <ResearchResult report={report} />
          </div>
        )}
      </div>
    </main>
  );
}
