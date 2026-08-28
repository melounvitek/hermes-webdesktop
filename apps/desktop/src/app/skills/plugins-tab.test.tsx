import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import { $pluginInstallRequest, closePluginInstallRequest } from '@/store/plugin-install-request'

import { PluginsTab } from './plugins-tab'

const requestGateway = vi.fn(async () => ({ plugins: [] }))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

describe('PluginsTab', () => {
  beforeEach(() => {
    $agentPlugins.set([])
    $agentPluginsStatus.set('ready')
    closePluginInstallRequest()
    requestGateway.mockClear()
  })

  afterEach(cleanup)

  it('lists the scoped profile agent plugins with toggles', () => {
    $agentPlugins.set([
      {
        description: 'A test plugin',
        key: 'demo-plugin',
        name: 'demo-plugin',
        source: 'git',
        status: 'enabled',
        version: '1.0.0'
      }
    ])

    render(<PluginsTab profile="workbot" />)

    expect(screen.getByText('demo-plugin')).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'demo-plugin' }).getAttribute('aria-checked')).toBe('true')
  })

  it('hides bundled plugins (managed from their own surfaces)', () => {
    $agentPlugins.set([
      {
        description: '',
        key: 'image_gen/fal',
        name: 'fal',
        source: 'bundled',
        status: 'enabled',
        version: ''
      }
    ])

    render(<PluginsTab profile={null} />)

    expect(screen.queryByText('fal')).toBeNull()
    expect(screen.getByText(/No agent plugins installed/)).toBeTruthy()
  })

  it('loads the plugin list scoped to the selected profile', () => {
    render(<PluginsTab profile="workbot" />)

    expect(requestGateway).toHaveBeenCalledWith(
      'plugins.manage',
      expect.objectContaining({ action: 'list', profile: 'workbot' })
    )
  })

  it('opens the dual-target install modal from a catalog pick message', async () => {
    render(<PluginsTab profile="workbot" />)

    window.dispatchEvent(
      new MessageEvent('message', {
        data: {
          name: 'weather-plugin',
          repo: 'https://github.com/example/weather-plugin',
          sha: 'a'.repeat(40),
          subdir: '',
          tier: 'community',
          type: 'hermes-plugin-pick'
        },
        origin: 'https://hermes-agent.nousresearch.com'
      })
    )

    await waitFor(() => {
      const request = $pluginInstallRequest.get()

      expect(request).not.toBeNull()
      expect(request?.catalogName).toBe('weather-plugin')
      expect(request?.repo).toBe('https://github.com/example/weather-plugin')
      expect(request?.profile).toBe('workbot')
      expect(request?.sha).toBe('a'.repeat(40))
    })
  })

  it('ignores pick messages from foreign origins', () => {
    render(<PluginsTab profile={null} />)

    window.dispatchEvent(
      new MessageEvent('message', {
        data: {
          name: 'evil-plugin',
          repo: 'https://github.com/evil/evil-plugin',
          type: 'hermes-plugin-pick'
        },
        origin: 'https://evil.example.com'
      })
    )

    expect($pluginInstallRequest.get()).toBeNull()
  })

  it('appends the subdir fragment for multi-plugin repos', async () => {
    render(<PluginsTab profile={null} />)

    window.dispatchEvent(
      new MessageEvent('message', {
        data: {
          name: 'nested-plugin',
          repo: 'https://github.com/example/plugins-monorepo',
          subdir: 'nested-plugin',
          type: 'hermes-plugin-pick'
        },
        origin: 'https://hermes-agent.nousresearch.com'
      })
    )

    await waitFor(() => {
      expect($pluginInstallRequest.get()?.repo).toBe(
        'https://github.com/example/plugins-monorepo#nested-plugin'
      )
    })
  })
})
