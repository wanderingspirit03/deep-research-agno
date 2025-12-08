import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow long-running API routes for deep research
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },
  // Increase serverless function timeout (for Vercel deployment)
  serverExternalPackages: [],
};

export default nextConfig;
