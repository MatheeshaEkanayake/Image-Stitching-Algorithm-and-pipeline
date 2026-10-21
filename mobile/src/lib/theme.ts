/**
 * Design tokens: a dark, camera-first palette with one vivid accent.
 * Key names are kept from the earlier blue theme so every screen picks up the new look;
 * `blueDark` is now the light accent used for text on tinted surfaces.
 */
export const colors = {
  blue: '#5B8CFF', // accent
  blueDark: '#B4C8FF', // accent text on tinted surfaces
  blueSoft: '#172138', // tinted surface
  blueBorder: '#2B3A5E',
  ink: '#05080F', // deepest background (camera, viewer)
  text: '#EEF2FB',
  muted: '#8B97B0',
  line: '#1E2740',
  bg: '#0A0F1A',
  card: '#121A2B',
  inputBg: '#0E1524',
  danger: '#FF6B6B',
  dangerSoft: 'rgba(255,107,107,0.14)',
  success: '#3DDC97',
  successSoft: 'rgba(61,220,151,0.14)',
  warning: '#FFB454',
  warningSoft: 'rgba(255,180,84,0.14)',
  white: '#FFFFFF',
  glass: 'rgba(8,12,22,0.62)', // camera overlays
  glassBorder: 'rgba(255,255,255,0.14)',
} as const;

export const radius = { sm: 10, md: 14, lg: 22, pill: 999 } as const;
export const space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24 } as const;

export const shadow = {
  shadowColor: '#000',
  shadowOpacity: 0.35,
  shadowRadius: 16,
  shadowOffset: { width: 0, height: 8 },
  elevation: 4,
} as const;
