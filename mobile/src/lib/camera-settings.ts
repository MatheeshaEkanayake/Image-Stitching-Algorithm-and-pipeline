import * as SecureStore from 'expo-secure-store';
import { useCallback, useEffect, useState } from 'react';

/** Which edge of the new frame shows the previous photo's overlap strip. */
export type GuideSide = 'right' | 'left' | 'down' | 'up';

export type CameraSettings = {
  grid: boolean;
  ruler: boolean;
  /** Tilting cross (level) in the middle of the screen. */
  level: boolean;
  /** Ghost of the previous photo to line the next shot up with. */
  guide: boolean;
  /** 0.15 to 1: how solid the ghost is (the earlier fixed value, 0.38, was too faint). */
  guideOpacity: number;
  guideSide: GuideSide;
};

export const DEFAULT_SETTINGS: CameraSettings = {
  grid: true,
  ruler: false,
  level: true,
  guide: true,
  guideOpacity: 0.7,
  guideSide: 'right',
};

const KEY = 'camera_settings';

function parse(raw: string | null): CameraSettings {
  if (!raw) return DEFAULT_SETTINGS;
  try {
    const value = JSON.parse(raw) as Partial<CameraSettings>;
    const opacity = typeof value.guideOpacity === 'number' ? Math.min(1, Math.max(0.15, value.guideOpacity)) : undefined;
    return { ...DEFAULT_SETTINGS, ...value, ...(opacity === undefined ? {} : { guideOpacity: opacity }) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

/** Camera overlay preferences, kept in secure storage so they survive restarts. */
export function useCameraSettings() {
  const [settings, setSettings] = useState<CameraSettings>(DEFAULT_SETTINGS);

  useEffect(() => {
    SecureStore.getItemAsync(KEY).then(
      (raw) => setSettings(parse(raw)),
      () => undefined,
    );
  }, []);

  const update = useCallback((changes: Partial<CameraSettings>) => {
    setSettings((current) => {
      const next = { ...current, ...changes };
      SecureStore.setItemAsync(KEY, JSON.stringify(next)).catch(() => undefined);
      return next;
    });
  }, []);

  return [settings, update] as const;
}
