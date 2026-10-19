/** Blue-highlighted design tokens used across the app. */
export const colors = {
  blue: '#1E6FFF',
  blueDark: '#0B4FD1',
  blueSoft: '#E8F0FF',
  blueBorder: '#B9CEFF',
  ink: '#0B1220',
  text: '#1B2433',
  muted: '#5D6B85',
  line: '#E3E8F2',
  bg: '#F5F8FF',
  card: '#FFFFFF',
  danger: '#D93636',
  dangerSoft: '#FDECEC',
  success: '#12924F',
  successSoft: '#E5F6ED',
  warning: '#B26A00',
  warningSoft: '#FFF4E0',
  white: '#FFFFFF',
} as const;

export const radius = { sm: 8, md: 12, lg: 18, pill: 999 } as const;
export const space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24 } as const;

export const shadow = {
  shadowColor: '#1E6FFF',
  shadowOpacity: 0.08,
  shadowRadius: 12,
  shadowOffset: { width: 0, height: 4 },
  elevation: 2,
} as const;
