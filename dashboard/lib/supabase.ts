import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

export const supabase = createClient(supabaseUrl, supabaseServiceKey)

export interface TrainingRun {
  id: string
  name: string
  algorithm: string
  environment: string
  status: string
  config: any
  metrics: any
  created_at: string
  updated_at: string
  completed_at?: string
}

export interface Rollout {
  id: string
  training_run_id?: string
  episode_number: number
  environment: string
  task_description?: string
  status: string
  total_reward: number
  step_count: number
  started_at: string
  completed_at?: string
  metadata: any
}

export interface AgentState {
  id: string
  rollout_id: string
  step_number: number
  observation: string
  action: string
  thought?: string
  memory_summary?: string
  created_at: string
}

export interface Metrics {
  total_rollouts: number
  completed_rollouts: number
  successful_rollouts: number
  success_rate: number
  average_reward: number
  average_steps: number
  rollouts: Rollout[]
}
