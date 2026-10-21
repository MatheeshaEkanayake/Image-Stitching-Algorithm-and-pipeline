import { Ionicons } from '@expo/vector-icons';
import { Accelerometer } from 'expo-sensors';
import { useEffect, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { colors, radius } from '@/lib/theme';

/** iPhone-style rule-of-thirds grid. A third is also the recommended overlap between shots. */
export function CameraGrid() {
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      {[1, 2].map((n) => (
        <View key={`v${n}`} style={[styles.line, styles.vertical, { left: `${(n * 100) / 3}%` }]} />
      ))}
      {[1, 2].map((n) => (
        <View key={`h${n}`} style={[styles.line, styles.horizontal, { top: `${(n * 100) / 3}%` }]} />
      ))}
      <View style={styles.centre} />
    </View>
  );
}

export type Level = {
  /** 'upright': aimed at a wall or scene; 'flat': looking straight down at a table. */
  mode: 'upright' | 'flat';
  /** Degrees off level: horizon roll when upright, tilt away from flat when flat. */
  error: number;
  /** Degrees between the screen and vertical (0 = upright, 90 = flat). */
  lean: number;
};

const LEVEL_TOLERANCE_DEG = 1.5;

/**
 * Phone orientation from gravity. Uses only magnitudes, so it does not depend on the
 * accelerometer's axis sign convention, which the Expo docs do not specify per platform.
 */
export function levelFromGravity(x: number, y: number, z: number): Level | null {
  const magnitude = Math.hypot(x, y, z);
  if (!Number.isFinite(magnitude) || magnitude < 0.5) return null; // no usable reading
  const deg = (rad: number) => (rad * 180) / Math.PI;
  const lean = deg(Math.asin(Math.min(1, Math.abs(z) / magnitude)));
  if (lean > 45) return { mode: 'flat', error: 90 - lean, lean };
  // angle of gravity from the nearest screen axis: 0 when the picture's horizon is level,
  // whether the phone is in portrait or landscape
  const roll = deg(Math.atan2(Math.min(Math.abs(x), Math.abs(y)), Math.max(Math.abs(x), Math.abs(y))));
  return { mode: 'upright', error: roll, lean };
}

/** Live level reading while ``active``; the sensor is only subscribed when needed. */
export function useLevel(active: boolean): Level | null {
  const [level, setLevel] = useState<Level | null>(null);
  useEffect(() => {
    if (!active) return;
    Accelerometer.setUpdateInterval(150);
    const subscription = Accelerometer.addListener(({ x, y, z }) => setLevel(levelFromGravity(x, y, z)));
    return () => subscription.remove();
  }, [active]);
  return active ? level : null;
}

export function LevelBadge({ level }: { level: Level | null }) {
  if (!level) return null;
  const ok = level.error <= LEVEL_TOLERANCE_DEG;
  const degrees = `${level.error.toFixed(0)}°`;
  const text = level.mode === 'flat' ? (ok ? 'Flat: level' : `Flat: tilted ${degrees}`) : ok ? 'Level' : `Tilted ${degrees}`;
  return (
    <View
      accessibilityLiveRegion="polite"
      accessibilityLabel={text}
      style={[styles.badge, { backgroundColor: ok ? colors.success : 'rgba(11,18,32,0.7)' }]}>
      <Ionicons name={ok ? 'checkmark-circle' : 'phone-portrait-outline'} size={14} color={colors.white} />
      <Text style={styles.badgeText}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  line: { position: 'absolute', backgroundColor: 'rgba(255,255,255,0.45)' },
  vertical: { top: 0, bottom: 0, width: StyleSheet.hairlineWidth * 2 },
  horizontal: { left: 0, right: 0, height: StyleSheet.hairlineWidth * 2 },
  centre: {
    position: 'absolute',
    left: '50%',
    top: '50%',
    width: 18,
    height: 18,
    marginLeft: -9,
    marginTop: -9,
    borderRadius: 9,
    borderWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.7)',
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: radius.pill,
  },
  badgeText: { color: colors.white, fontWeight: '700', fontSize: 13 },
});
