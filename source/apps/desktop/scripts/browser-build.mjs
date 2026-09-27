import { execFileSync } from 'node:child_process'

export function getBrowserBuild(repoRoot) {
  const git = args =>
    execFileSync('git', args, {
      cwd: repoRoot,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe']
    }).trim()

  try {
    const revision = git(['rev-parse', '--verify', 'HEAD'])
    const status = git([
      'status',
      '--porcelain',
      '--untracked-files=all',
      '--',
      'apps/desktop',
      'apps/shared',
      ':(glob)package*.json'
    ])
    return { revision, dirty: status.length > 0 }
  } catch {
    return { revision: null, dirty: null }
  }
}
