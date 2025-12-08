'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Download, FileText, Share2, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState } from 'react';

interface ResearchResultProps {
  report: string;
}

export function ResearchResult({ report }: ResearchResultProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(report);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research-report-${new Date().toISOString().slice(0, 10)}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 animate-in fade-in duration-700">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="px-3 py-1 text-sm bg-green-500/10 text-green-600 border-green-200">
            <FileText className="h-3 w-3 mr-1" />
            Research Complete
          </Badge>
          <span className="text-sm text-muted-foreground">
            {report.length > 0 ? `${Math.round(report.length / 5)} words` : 'Processing...'}
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={handleCopy} title="Copy to clipboard">
            {copied ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleDownload} title="Download Markdown">
            <Download className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="sm" title="Share">
            <Share2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Card className="border-0 shadow-lg bg-card/50 backdrop-blur-sm overflow-hidden">
        <ScrollArea className="h-[calc(100vh-200px)] w-full px-6 py-8">
          <div className="prose prose-slate dark:prose-invert max-w-none prose-headings:font-semibold prose-h1:text-3xl prose-h2:text-2xl prose-a:text-blue-500 hover:prose-a:underline">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {report}
            </ReactMarkdown>
          </div>
        </ScrollArea>
      </Card>
    </div>
  );
}
