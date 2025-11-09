import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs'
import { cookies } from 'next/headers'
import { NextResponse } from 'next/server'

export async function GET(request: Request) {
  const requestUrl = new URL(request.url)
  const { searchParams } = requestUrl

  const trainingRunId = searchParams.get('training_run_id')
  const environment = searchParams.get('environment')
  const limit = parseInt(searchParams.get('limit') || '100')

  const supabase = createRouteHandlerClient({ cookies })

  try {
    // Build query
    let query = supabase
      .from('rollouts')
      .select(`
        id,
        episode_number,
        environment,
        status,
        total_reward,
        step_count,
        started_at,
        completed_at,
        training_run_id
      `)
      .order('started_at', { ascending: false })
      .limit(limit)

    if (trainingRunId) {
      query = query.eq('training_run_id', trainingRunId)
    }

    if (environment) {
      query = query.eq('environment', environment)
    }

    const { data: rollouts, error } = await query

    if (error) {
      return NextResponse.json(
        { error: 'Failed to fetch metrics', details: error.message },
        { status: 500 }
      )
    }

    // Calculate statistics
    const completed = rollouts.filter((r: any) => r.status !== 'running')
    const successful = rollouts.filter((r: any) => r.status === 'success')

    const avgReward =
      completed.length > 0
        ? completed.reduce((sum: number, r: any) => sum + (r.total_reward || 0), 0) /
          completed.length
        : 0

    const avgSteps =
      completed.length > 0
        ? completed.reduce((sum: number, r: any) => sum + (r.step_count || 0), 0) /
          completed.length
        : 0

    const successRate = completed.length > 0 ? successful.length / completed.length : 0

    const metrics = {
      total_rollouts: rollouts.length,
      completed_rollouts: completed.length,
      successful_rollouts: successful.length,
      success_rate: successRate,
      average_reward: avgReward,
      average_steps: avgSteps,
      rollouts: rollouts.slice(0, 20),
    }

    return NextResponse.json(metrics)
  } catch (error: any) {
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
