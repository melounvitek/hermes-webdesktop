export const BROWSER_BUILD =
  typeof __HERMES_BROWSER_BUILD__ === 'undefined' ? { revision: null, dirty: null } : __HERMES_BROWSER_BUILD__
