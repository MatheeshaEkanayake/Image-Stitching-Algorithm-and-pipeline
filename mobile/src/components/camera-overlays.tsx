import { Accelerometer } from 'expo-sensors';
import { useEffect, useRef, useState } from 'react';
import { Animated, Platform, StyleSheet, View } from 'react-native';

import { colors } from '@/lib/theme';

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
    </View>
  );
}

const TICKS = 20; // one tick per 5% of the frame

/** Tick marks along the top and left edges, with a longer mark every 10% and the centre. */
export function CameraRuler() {
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      {Array.from({ length: TICKS + 1 }, (_, i) => {
        const major = i % 2 === 0;
        const centre = i === TICKS / 2;
        const length = centre ? 26 : major ? 16 : 9;
        const pos = `${(i * 100) / TICKS}%` as const;
        return (
          <View key={i}>
            <View style={[styles.tick, { top: 0, left: pos, width: 1.5, height: length }]} />
            <View style={[styles.tick, { left: 0, top: pos, height: 1.5, width: length }]} />
          </View>
        );
      })}
    </View>
  );
}

/** Degrees off level on each axis: positive x = tilted right, positive y = tilted down. */
export type Tilt = {
  /** 'upright': aimed at a wall or scene; 'flat': looking straight down at a table. */
  mode: 'upright' | 'flat';
  x: number;
  y: number;
};

const LEVEL_TOLERANCE_DEG = 1.5;
const deg = (rad: number) => (rad * 180) / Math.PI;

/**
 * Phone orientation from gravity. expo-sensors reports opposite signs on iOS and Android, so
 * the reading is first brought to the Android convention (resting face up gives z = +1,
 * held upright in portrait gives y = +1).
 */
export function tiltFromGravity(rawX: number, rawY: number, rawZ: number): Tilt | null {
  const sign = Platform.OS === 'ios' ? -1 : 1;
  const x = rawX * sign;
  const y = rawY * sign;
  const z = rawZ * sign;
  const magnitude = Math.hypot(x, y, z);
  if (!Number.isFinite(magnitude) || magnitude < 0.5) return null; // no usable reading
  const lean = deg(Math.asin(Math.min(1, Math.abs(z) / magnitude)));
  if (lean > 45) {
    // shooting down at a table: both axes are tilts away from flat
    return { mode: 'flat', x: deg(Math.asin(Math.max(-1, Math.min(1, x / magnitude)))), y: deg(Math.asin(Math.max(-1, Math.min(1, y / magnitude)))) };
  }
  // held up at a scene: horizontal = roll of the horizon, vertical = pitch up/down
  return { mode: 'upright', x: deg(Math.atan2(x, Math.abs(y))), y: deg(Math.asin(Math.max(-1, Math.min(1, z / magnitude)))) };
}

/** Live tilt reading while ``active``; the sensor is only subscribed when needed. */
export function useTilt(active: boolean, onTilt: (tilt: Tilt | null) => void) {
  const callback = useRef(onTilt);
  callback.current = onTilt;
  useEffect(() => {
    if (!active) return;
    Accelerometer.setUpdateInterval(50);
    const subscription = Accelerometer.addListener(({ x, y, z }) => callback.current(tiltFromGravity(x, y, z)));
    return () => subscription.remove();
  }, [active]);
}

const PX_PER_DEG = 7;
const MAX_OFFSET = 80;
const clamp = (n: number) => Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, n));

/**
 * Level indicator like the iPhone camera's: a fixed cross in the middle of the screen and a
 * second cross that slides horizontally and vertically with the phone's tilt. When both are
 * on top of each other (within a degree or two) the cross turns green.
 */
export function LevelCross({ active }: { active: boolean }) {
  const offset = useRef(new Animated.ValueXY({ x: 0, y: 0 })).current;
  const [aligned, setAligned] = useState(false);
  const alignedRef = useRef(false);

  useTilt(active, (tilt) => {
    if (!tilt) return;
    // the cross moves the way the phone leans, like a bubble drifting to the high side
    offset.setValue({ x: clamp(tilt.x * PX_PER_DEG), y: clamp(-tilt.y * PX_PER_DEG) });
    const ok = Math.abs(tilt.x) <= LEVEL_TOLERANCE_DEG && Math.abs(tilt.y) <= LEVEL_TOLERANCE_DEG;
    if (ok !== alignedRef.current) {
      alignedRef.current = ok;
      setAligned(ok);
    }
  });

  if (!active) return null;
  const tint = aligned ? colors.success : '#FFD60A';
  return (
    <View pointerEvents="none" style={styles.crossHost} accessibilityLabel={aligned ? 'Level' : 'Not level'}>
      {/* fixed reference cross */}
      <Cross color="rgba(255,255,255,0.85)" />
      {/* moving cross, driven by the accelerometer */}
      <Animated.View style={[styles.crossLayer, { transform: offset.getTranslateTransform() }]}>
        <Cross color={tint} thick />
      </Animated.View>
    </View>
  );
}

function Cross({ color, thick }: { color: string; thick?: boolean }) {
  const t = thick ? 2.5 : 1.5;
  return (
    <View style={styles.crossLayer}>
      <View style={{ position: 'absolute', width: 34, height: t, backgroundColor: color, borderRadius: t }} />
      <View style={{ position: 'absolute', width: t, height: 34, backgroundColor: color, borderRadius: t }} />
    </View>
  );
}

const styles = StyleSheet.create({
  line: { position: 'absolute', backgroundColor: 'rgba(255,255,255,0.4)' },
  vertical: { top: 0, bottom: 0, width: StyleSheet.hairlineWidth * 2 },
  horizontal: { left: 0, right: 0, height: StyleSheet.hairlineWidth * 2 },
  tick: { position: 'absolute', backgroundColor: 'rgba(255,214,10,0.9)' },
  crossHost: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center' },
  crossLayer: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center' },
});
