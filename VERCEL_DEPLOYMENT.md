# 🚀 OpenManus-RL Vercel Deployment Guide

## Prerequisites

- GitHub account
- Vercel account (sign up at https://vercel.com)
- Supabase project (already configured)

## Quick Deploy to Vercel

### Option 1: Deploy Button (Fastest)

Click the button below to deploy directly to Vercel:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fu4231458123-droid%2Fnexifyai-openmanus&project-name=openmanus-rl-dashboard&repository-name=openmanus-rl-dashboard&env=NEXT_PUBLIC_SUPABASE_URL,NEXT_PUBLIC_SUPABASE_ANON_KEY,SUPABASE_SERVICE_ROLE_KEY)

### Option 2: Manual Deployment

1. **Push to GitHub** (already done ✅)
   ```bash
   git push origin main
   ```

2. **Connect to Vercel**
   - Go to https://vercel.com/new
   - Import your repository: `u4231458123-droid/nexifyai-openmanus`
   - Select the repository

3. **Configure Project**
   - Framework Preset: **Next.js**
   - Root Directory: `dashboard`
   - Build Command: `npm run build`
   - Output Directory: `.next`

4. **Set Environment Variables**

   Add these in Vercel's Environment Variables section:

   ```
   NEXT_PUBLIC_SUPABASE_URL=https://jdjhkmenfkmbaeaskkug.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=sb_publishable_IJFhatPZZcKJfB8G5QC9Tg_TqP4nTcX
   SUPABASE_SERVICE_ROLE_KEY=sbp_ed71b8e9dd2c7d7205d626b99ad63a218934e67c
   ```

5. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete (~2-3 minutes)
   - Your dashboard will be live!

## Local Development

```bash
# Navigate to dashboard directory
cd dashboard

# Install dependencies
npm install

# Run development server
npm run dev

# Open http://localhost:3000
```

## Project Structure

```
dashboard/
├── app/
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Main dashboard page
│   ├── globals.css         # Global styles
│   └── api/
│       └── metrics/
│           └── route.ts    # Metrics API endpoint
├── lib/
│   ├── supabase.ts         # Supabase client & types
│   └── utils.ts            # Utility functions
├── package.json            # Dependencies
├── next.config.js          # Next.js configuration
├── tailwind.config.js      # Tailwind CSS config
├── tsconfig.json           # TypeScript config
└── .env.local              # Local environment variables
```

## Features

✅ **Real-time Dashboard**
- Training runs overview
- Rollout metrics and statistics
- Success rate tracking
- Average reward and step count

✅ **Modern UI**
- Dark mode by default
- Responsive design
- Tailwind CSS styling
- Lucide icons

✅ **Performance**
- Server-side rendering (SSR)
- Static generation where possible
- Optimized for Vercel Edge Network

✅ **API Endpoints**
- `/api/metrics` - Get training metrics
- Supabase integration for data

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NEXT_PUBLIC_SUPABASE_URL` | Your Supabase project URL | Yes |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase anonymous key | Yes |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key (server-side) | Yes |

## Vercel Configuration

The `vercel.json` file includes:
- Build configuration
- API route handling
- CORS headers
- Environment variable mapping

## Post-Deployment

After deploying to Vercel:

1. **Verify the deployment**
   - Visit your Vercel dashboard URL
   - Check all metrics are loading

2. **Configure custom domain** (optional)
   - Go to Vercel project settings
   - Add your custom domain
   - Update DNS records

3. **Enable analytics** (optional)
   - Vercel Analytics provides insights
   - Go to project settings > Analytics

4. **Set up monitoring**
   - Use Vercel's built-in monitoring
   - Check logs for errors

## Troubleshooting

### Build Fails

If build fails, check:
- All environment variables are set
- Node.js version compatibility (v18+ recommended)
- Review build logs in Vercel dashboard

### Data Not Loading

If dashboard shows no data:
- Verify Supabase credentials
- Check Supabase RLS policies
- Ensure tables have data

### API Errors

If API endpoints fail:
- Check environment variables
- Verify Supabase service role key
- Review API logs in Vercel

## Automatic Deployments

Once connected to Vercel:
- Every push to `main` triggers a production deployment
- Pull requests create preview deployments
- Instant rollback available in Vercel dashboard

## Performance Tips

1. **Edge Functions**: API routes run on Vercel Edge Network
2. **ISR**: Use Incremental Static Regeneration for data that changes infrequently
3. **CDN**: Static assets served from Vercel's global CDN
4. **Caching**: Configure appropriate cache headers

## Next Steps

1. ✅ Deploy to Vercel
2. Configure custom domain
3. Set up monitoring and alerts
4. Add more dashboard features
5. Integrate with training scripts

## Support

- [Vercel Documentation](https://vercel.com/docs)
- [Next.js Documentation](https://nextjs.org/docs)
- [Supabase Documentation](https://supabase.com/docs)

---

**Your dashboard is ready to deploy! 🚀**

Simply push to GitHub and connect to Vercel to go live in minutes.
