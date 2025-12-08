'use client';

import { useState } from 'react';
import { ArrowRight, Search, Loader2 } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';

interface ResearchInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
}

export function ResearchInput({ onSubmit, isLoading }: ResearchInputProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit(query.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="space-y-2 text-center">
        <h1 className="text-4xl font-bold tracking-tight text-foreground sm:text-5xl">
          Deep Research
        </h1>
        <p className="text-muted-foreground text-lg">
          Ask any question to start a comprehensive analysis
        </p>
      </div>

      <div className="relative group">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-teal-500 rounded-xl blur-lg opacity-10 group-hover:opacity-20 transition-opacity" />
        <div className="relative bg-card rounded-xl border shadow-sm p-4">
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="E.g., What is the current state of solid-state battery technology?"
              className="min-h-[120px] resize-none border-0 bg-transparent text-lg focus-visible:ring-0 p-2 shadow-none"
              disabled={isLoading}
            />
            
            <div className="flex justify-between items-center border-t pt-4">
              <div className="text-xs text-muted-foreground flex items-center gap-2">
                <Search className="h-4 w-4" />
                <span>PhD-level research depth</span>
              </div>
              <Button 
                type="submit" 
                size="icon"
                disabled={!query.trim() || isLoading}
                className="rounded-full w-10 h-10 transition-all hover:scale-105"
              >
                {isLoading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <ArrowRight className="h-5 w-5" />
                )}
                <span className="sr-only">Start Research</span>
              </Button>
            </div>
          </form>
        </div>
      </div>
      
      <div className="flex flex-wrap gap-2 justify-center">
        {['Market Analysis', 'Technology Trends', 'Competitive Landscape', 'Academic Literature'].map((tag) => (
          <button
            key={tag}
            onClick={() => setQuery(tag)}
            className="text-xs px-3 py-1.5 rounded-full bg-secondary/50 hover:bg-secondary text-muted-foreground transition-colors"
          >
            {tag}
          </button>
        ))}
      </div>
    </div>
  );
}
