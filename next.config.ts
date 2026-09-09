import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'standalone',
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
  async rewrites() {
    return {
      beforeFiles: [
        {
          source: '/:path*',
          has: [
            {
              type: 'query',
              key: 'XTransformPort',
            },
          ],
          destination: `${process.env.BACKEND_URL || 'http://127.0.0.1:3003'}/:path*`,
        },
      ],
    };
  },
};

export default nextConfig;
