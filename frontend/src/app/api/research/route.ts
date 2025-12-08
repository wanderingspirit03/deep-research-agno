import { NextRequest, NextResponse } from 'next/server';

// Allow very long research times (up to 2 hours)
export const maxDuration = 7200;

export async function POST(req: NextRequest) {
  try {
    const { query } = await req.json();
    
    // Default to localhost:8000 if not specified
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    
    const response = await fetch(
      `${backendUrl}/api/research`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
        // No timeout - allow research to run as long as needed
      }
    );
    
    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { error: `Backend error: ${response.status} ${errorText}` },
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Research API Error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal Server Error' },
      { status: 500 }
    );
  }
}
