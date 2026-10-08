const OPT_OUT = {
  autocapitalize: 'off',
  autocomplete: 'off',
  autocorrect: 'off',
  spellcheck: 'false',
  'data-1p-ignore': 'true',
  'data-bwignore': 'true',
  'data-form-type': 'other',
  'data-lpignore': 'true'
}

function optOut(input: Element) {
  for (const [name, value] of Object.entries(OPT_OUT)) {
    input.setAttribute(name, value)
  }
}

// The app's secret fields (sudo, vault and API keys) are never logins to the
// Hermes site, yet browsers and password managers offer to save or fill them.
// Chrome and Firefox treat an input as a password field once it has been one,
// so `type="password"` becomes a masked text field before the input is attached.
// Without text masking support it stays a password field with opt-out hints.
export function quietPasswordFields() {
  const type = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'type')!
  const mask = CSS.supports('-webkit-text-security', 'disc')

  if (mask) {
    const style = document.createElement('style')
    style.textContent = 'input[data-hermes-masked] { -webkit-text-security: disc; }'
    document.head.appendChild(style)
  }

  // React sets this property whenever it creates or updates an input.
  Object.defineProperty(HTMLInputElement.prototype, 'type', {
    ...type,
    set(this: HTMLInputElement, value: string) {
      const secret = value === 'password'

      if (secret) {
        optOut(this)
      }

      type.set!.call(this, secret && mask ? 'text' : value)
      this.toggleAttribute('data-hermes-masked', secret && mask)
    }
  })

  // The website login prompt's username field would still offer saved logins.
  new MutationObserver(() => document.querySelectorAll('input[autocomplete="username"]').forEach(optOut)).observe(
    document.documentElement,
    { childList: true, subtree: true }
  )
}
