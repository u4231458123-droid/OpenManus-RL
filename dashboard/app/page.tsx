import { supabase, type Metrics, type TrainingRun, type Rollout } from '@/lib/supabase'
import { formatDate, formatReward, getStatusBadge } from '@/lib/utils'
import { Activity, TrendingUp, CheckCircle2, XCircle, Clock } from 'lucide-react'
import Link from 'next/link'

async function getMetrics(): Promise<Metrics> {
  const { data: rollouts, error } = await supabase
    .from('rollouts')
    .select('*')
    .order('started_at', { ascending: false })
    .limit(100)

  if (error) {
    console.error('Error fetching metrics:', error)
    return {
      total_rollouts: 0,
      completed_rollouts: 0,
      successful_rollouts: 0,
      success_rate: 0,
      average_reward: 0,
      average_steps: 0,
      rollouts: [],
    }
  }

  const completed = rollouts.filter((r: any) => r.status !== 'running')
  const successful = rollouts.filter((r: any) => r.status === 'success')

  return {
    total_rollouts: rollouts.length,
    completed_rollouts: completed.length,
    successful_rollouts: successful.length,
    success_rate: completed.length > 0 ? successful.length / completed.length : 0,
    average_reward:
      completed.length > 0
        ? completed.reduce((sum: number, r: any) => sum + (r.total_reward || 0), 0) /
          completed.length
        : 0,
    average_steps:
      completed.length > 0
        ? completed.reduce((sum: number, r: any) => sum + (r.step_count || 0), 0) /
          completed.length
        : 0,
    rollouts: rollouts.slice(0, 20),
  }
}

async function getTrainingRuns(): Promise<TrainingRun[]> {
  const { data, error } = await supabase
    .from('training_runs')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(10)

  if (error) {
    console.error('Error fetching training runs:', error)
    return []
  }

  return data || []
}

export default async function Dashboard() {
  const metrics = await getMetrics()
  const trainingRuns = await getTrainingRuns()

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Header */}
      <header className="border-b border-gray-700 bg-gray-900/50 backdrop-blur">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white">OpenManus-RL</h1>
              <p className="text-gray-400 mt-1">Training Dashboard & Analytics</p>
            </div>
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-2 px-3 py-1 bg-green-500/10 border border-green-500/20 rounded-full text-green-500 text-sm">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                Live
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Total Rollouts"
            value={metrics.total_rollouts}
            icon={<Activity className="w-6 h-6" />}
            color="blue"
          />
          <StatCard
            title="Success Rate"
            value={`${(metrics.success_rate * 100).toFixed(1)}%`}
            icon={<TrendingUp className="w-6 h-6" />}
            color="green"
          />
          <StatCard
            title="Avg Reward"
            value={formatReward(metrics.average_reward)}
            icon={<CheckCircle2 className="w-6 h-6" />}
            color="purple"
          />
          <StatCard
            title="Avg Steps"
            value={Math.round(metrics.average_steps)}
            icon={<Clock className="w-6 h-6" />}
            color="orange"
          />
        </div>

        {/* Training Runs */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">Recent Training Runs</h2>
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-900/50 border-b border-gray-700">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Algorithm
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Environment
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Created
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700">
                  {trainingRuns.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="px-6 py-8 text-center text-gray-400">
                        No training runs yet. Start your first experiment!
                      </td>
                    </tr>
                  ) : (
                    trainingRuns.map((run) => (
                      <tr key={run.id} className="hover:bg-gray-700/30 transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                          {run.name}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                          {run.algorithm.toUpperCase()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                          {run.environment}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span
                            className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full border ${getStatusBadge(
                              run.status
                            )}`}
                          >
                            {run.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                          {formatDate(run.created_at)}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Recent Rollouts */}
        <div>
          <h2 className="text-2xl font-bold text-white mb-4">Recent Rollouts</h2>
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-900/50 border-b border-gray-700">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Episode
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Environment
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Reward
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Steps
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Started
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700">
                  {metrics.rollouts.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="px-6 py-8 text-center text-gray-400">
                        No rollouts yet. Run your first episode!
                      </td>
                    </tr>
                  ) : (
                    metrics.rollouts.map((rollout) => (
                      <tr key={rollout.id} className="hover:bg-gray-700/30 transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                          #{rollout.episode_number}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                          {rollout.environment}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span
                            className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full border ${getStatusBadge(
                              rollout.status
                            )}`}
                          >
                            {rollout.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                          {formatReward(rollout.total_reward)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                          {rollout.step_count}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                          {formatDate(rollout.started_at)}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-700 mt-12">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <div>
              Powered by{' '}
              <a
                href="https://supabase.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-green-500 hover:text-green-400"
              >
                Supabase
              </a>{' '}
              &{' '}
              <a
                href="https://vercel.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 hover:text-blue-400"
              >
                Vercel
              </a>
            </div>
            <div>OpenManus-RL Dashboard v1.0</div>
          </div>
        </div>
      </footer>
    </div>
  )
}

function StatCard({
  title,
  value,
  icon,
  color,
}: {
  title: string
  value: string | number
  icon: React.ReactNode
  color: 'blue' | 'green' | 'purple' | 'orange'
}) {
  const colorClasses = {
    blue: 'from-blue-500/20 to-blue-600/20 border-blue-500/30',
    green: 'from-green-500/20 to-green-600/20 border-green-500/30',
    purple: 'from-purple-500/20 to-purple-600/20 border-purple-500/30',
    orange: 'from-orange-500/20 to-orange-600/20 border-orange-500/30',
  }

  const iconColorClasses = {
    blue: 'text-blue-500',
    green: 'text-green-500',
    purple: 'text-purple-500',
    orange: 'text-orange-500',
  }

  return (
    <div
      className={`bg-gradient-to-br ${colorClasses[color]} backdrop-blur border rounded-lg p-6`}
    >
      <div className="flex items-center justify-between mb-4">
        <span className="text-gray-400 text-sm font-medium">{title}</span>
        <div className={iconColorClasses[color]}>{icon}</div>
      </div>
      <div className="text-3xl font-bold text-white">{value}</div>
    </div>
  )
}
