# OpenManus-RL Dashboard

Modern, real-time dashboard for monitoring OpenManus-RL training runs and rollouts.

## Features

- 📊 Real-time training metrics
- 📈 Success rate tracking
- 🎯 Rollout visualization
- 🚀 Built with Next.js 14 & Supabase
- 🎨 Beautiful UI with Tailwind CSS
- ⚡ Deployed on Vercel Edge Network

## Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

## Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fu4231458123-droid%2Fnexifyai-openmanus)

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **Database**: Supabase (PostgreSQL)
- **Deployment**: Vercel
- **Icons**: Lucide React
- **Charts**: Recharts

## License

Apache 2.0
