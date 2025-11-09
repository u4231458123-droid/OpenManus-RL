import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(date: string | Date) {
  return new Date(date).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatDuration(startDate: string, endDate?: string) {
  const start = new Date(startDate)
  const end = endDate ? new Date(endDate) : new Date()
  const duration = end.getTime() - start.getTime()

  const seconds = Math.floor(duration / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`
  } else {
    return `${seconds}s`
  }
}

export function formatReward(reward: number) {
  return reward.toFixed(2)
}

export function getStatusColor(status: string) {
  switch (status) {
    case 'success':
      return 'text-green-500'
    case 'failed':
      return 'text-red-500'
    case 'running':
      return 'text-blue-500'
    case 'timeout':
      return 'text-yellow-500'
    default:
      return 'text-gray-500'
  }
}

export function getStatusBadge(status: string) {
  switch (status) {
    case 'success':
      return 'bg-green-500/10 text-green-500 border-green-500/20'
    case 'failed':
      return 'bg-red-500/10 text-red-500 border-red-500/20'
    case 'running':
      return 'bg-blue-500/10 text-blue-500 border-blue-500/20'
    case 'timeout':
      return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20'
    default:
      return 'bg-gray-500/10 text-gray-500 border-gray-500/20'
  }
}
