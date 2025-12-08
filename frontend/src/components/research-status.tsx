'use client';

import { Loader2 } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { useState, useEffect } from 'react';

export function ResearchStatus() {
  const [elapsed, setElapsed] = useState(0);
  
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins === 0) return `${secs}s`;
    return `${mins}m ${secs}s`;
  };

  const steps = [
    'Analyzing query intent...',
    'Planning research strategy...',
    'Executing parallel searches...',
    'Evaluating source quality...',
    'Synthesizing findings...',
    'Generating comprehensive report...'
  ];

  return (
    <div className="w-full max-w-2xl mx-auto mt-12 animate-in fade-in duration-500">
      <Card className="p-8 border-0 shadow-lg bg-card/50 backdrop-blur-sm flex flex-col items-center text-center gap-6">
        <div className="relative">
          <div className="absolute inset-0 bg-blue-500 blur-xl opacity-20 animate-pulse" />
          <Loader2 className="h-12 w-12 animate-spin text-primary relative z-10" />
        </div>
        
        <div className="space-y-2">
          <h3 className="text-xl font-semibold">Conducting Deep Research</h3>
          <p className="text-muted-foreground">
            PhD-level analysis in progress • {formatTime(elapsed)}
          </p>
          <p className="text-xs text-muted-foreground/70">
            Deep research typically takes 5-30 minutes depending on complexity
          </p>
        </div>

        <div className="w-full max-w-md space-y-3 pt-4">
          {steps.map((step, i) => (
            <div 
              key={i} 
              className="flex items-center gap-3 text-sm text-muted-foreground"
              style={{ opacity: 1 - i * 0.12 }}
            >
              <div className="h-1.5 w-1.5 rounded-full bg-primary/50 animate-pulse" />
              <span>{step}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
