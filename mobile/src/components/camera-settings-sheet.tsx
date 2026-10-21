import { Ionicons } from '@expo/vector-icons';
import { useRef, useState, type ComponentProps } from 'react';
import { Modal, PanResponder, Pressable, StyleSheet, Switch, Text, View, type GestureResponderEvent } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import type { CameraSettings, GuideSide } from '@/lib/camera-settings';
import { colors, radius, space } from '@/lib/theme';

const MIN_OPACITY = 0.15;
const SIDES: { key: GuideSide; label: string; icon: 'arrow-forward' | 'arrow-back' | 'arrow-down' | 'arrow-up' }[] = [
  { key: 'right', label: 'Right', icon: 'arrow-forward' },
  { key: 'left', label: 'Left', icon: 'arrow-back' },
  { key: 'down', label: 'Down', icon: 'arrow-down' },
  { key: 'up', label: 'Up', icon: 'arrow-up' },
];

type Props = {
  visible: boolean;
  settings: CameraSettings;
  onChange: (changes: Partial<CameraSettings>) => void;
  onClose: () => void;
};

/** Bottom sheet with the capture overlay options. */
export function CameraSettingsSheet({ visible, settings, onChange, onClose }: Props) {
  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose} statusBarTranslucent>
      <Pressable style={styles.backdrop} onPress={onClose} accessibilityLabel="Close settings" />
      <SafeAreaView edges={['bottom']} style={styles.sheet}>
        <View style={styles.handle} />
        <View style={styles.titleRow}>
          <Text style={styles.title}>Capture settings</Text>
          <Pressable onPress={onClose} hitSlop={12} accessibilityRole="button" accessibilityLabel="Done">
            <Text style={styles.done}>Done</Text>
          </Pressable>
        </View>

        <ToggleRow icon="grid-outline" label="Grid" hint="Rule-of-thirds lines" value={settings.grid} onChange={(grid) => onChange({ grid })} />
        <ToggleRow icon="resize-outline" label="Ruler" hint="Tick marks along the top and left edges" value={settings.ruler} onChange={(ruler) => onChange({ ruler })} />
        <ToggleRow icon="add-outline" label="Level cross" hint="Tilting cross like the iPhone camera" value={settings.level} onChange={(level) => onChange({ level })} />
        <ToggleRow icon="layers-outline" label="Overlap guide" hint="Ghost of your previous photo" value={settings.guide} onChange={(guide) => onChange({ guide })} />

        {settings.guide ? (
          <View style={styles.group}>
            <View style={styles.sliderHead}>
              <Text style={styles.label}>Guide opacity</Text>
              <Text style={styles.value}>{Math.round(settings.guideOpacity * 100)}%</Text>
            </View>
            <Slider value={settings.guideOpacity} min={MIN_OPACITY} max={1} onChange={(guideOpacity) => onChange({ guideOpacity })} />

            <Text style={[styles.label, { marginTop: space.md }]}>Where the next photo goes</Text>
            <View style={styles.segment}>
              {SIDES.map((s) => {
                const active = settings.guideSide === s.key;
                return (
                  <Pressable
                    key={s.key}
                    accessibilityRole="button"
                    accessibilityState={{ selected: active }}
                    onPress={() => onChange({ guideSide: s.key })}
                    style={[styles.segmentItem, active && styles.segmentActive]}>
                    <Ionicons name={s.icon} size={16} color={active ? colors.white : colors.muted} />
                    <Text style={[styles.segmentText, active && { color: colors.white }]}>{s.label}</Text>
                  </Pressable>
                );
              })}
            </View>
            <Text style={styles.hint}>Pick the direction you will move the phone next; the guide shows the edge of your last photo to line up with.</Text>
          </View>
        ) : null}
      </SafeAreaView>
    </Modal>
  );
}

function ToggleRow({
  icon,
  label,
  hint,
  value,
  onChange,
}: {
  icon: ComponentProps<typeof Ionicons>['name'];
  label: string;
  hint: string;
  value: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <View style={styles.row}>
      <View style={styles.rowIcon}>
        <Ionicons name={icon} size={18} color={colors.blueDark} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={styles.rowLabel}>{label}</Text>
        <Text style={styles.rowHint}>{hint}</Text>
      </View>
      <Switch
        value={value}
        onValueChange={onChange}
        accessibilityLabel={label}
        trackColor={{ true: colors.blue, false: colors.line }}
        thumbColor={colors.white}
      />
    </View>
  );
}

/** Small drag-or-tap slider (the core library has none). */
function Slider({ value, min, max, onChange }: { value: number; min: number; max: number; onChange: (value: number) => void }) {
  const [width, setWidth] = useState(0);
  const widthRef = useRef(0);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const setFrom = (e: GestureResponderEvent) => {
    if (!widthRef.current) return;
    const ratio = Math.min(1, Math.max(0, e.nativeEvent.locationX / widthRef.current));
    onChangeRef.current(Math.round((min + ratio * (max - min)) * 100) / 100);
  };
  const pan = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderTerminationRequest: () => false, // keep the drag even inside a modal
      onPanResponderGrant: setFrom,
      onPanResponderMove: setFrom,
    }),
  ).current;

  const ratio = (value - min) / (max - min);
  return (
    <View
      {...pan.panHandlers}
      accessibilityRole="adjustable"
      accessibilityValue={{ min: 0, max: 100, now: Math.round(value * 100) }}
      onLayout={(e) => {
        widthRef.current = e.nativeEvent.layout.width;
        setWidth(e.nativeEvent.layout.width);
      }}
      style={styles.sliderHit}>
      <View pointerEvents="none" style={styles.track}>
        <View style={[styles.fill, { width: `${ratio * 100}%` }]} />
      </View>
      <View pointerEvents="none" style={[styles.thumb, { left: Math.max(0, ratio * width - 14) }]} />
    </View>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.45)' },
  sheet: {
    backgroundColor: colors.card,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    paddingHorizontal: space.xl,
    paddingTop: space.sm,
    paddingBottom: space.md,
    borderTopWidth: 1,
    borderColor: colors.blueBorder,
  },
  handle: { alignSelf: 'center', width: 40, height: 4, borderRadius: 2, backgroundColor: colors.blueBorder, marginBottom: space.md },
  titleRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: space.sm },
  title: { fontSize: 20, fontWeight: '800', color: colors.text },
  done: { fontSize: 16, fontWeight: '700', color: colors.blue },
  row: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: 10 },
  rowIcon: { width: 36, height: 36, borderRadius: 12, backgroundColor: colors.blueSoft, alignItems: 'center', justifyContent: 'center' },
  rowLabel: { fontSize: 16, fontWeight: '700', color: colors.text },
  rowHint: { fontSize: 12, color: colors.muted, marginTop: 1 },
  group: { marginTop: space.sm, padding: space.lg, borderRadius: radius.lg, backgroundColor: colors.inputBg, borderWidth: 1, borderColor: colors.line },
  sliderHead: { flexDirection: 'row', justifyContent: 'space-between' },
  label: { fontSize: 12, fontWeight: '700', color: colors.muted, textTransform: 'uppercase', letterSpacing: 0.8 },
  value: { fontSize: 14, fontWeight: '800', color: colors.text },
  sliderHit: { height: 40, justifyContent: 'center' },
  track: { height: 6, borderRadius: 3, backgroundColor: colors.line, overflow: 'hidden' },
  fill: { height: '100%', backgroundColor: colors.blue },
  thumb: { position: 'absolute', width: 28, height: 28, borderRadius: 14, backgroundColor: colors.white, borderWidth: 3, borderColor: colors.blue },
  segment: { flexDirection: 'row', gap: 6, marginTop: space.sm },
  segmentItem: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    paddingVertical: 10,
    borderRadius: radius.pill,
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.line,
  },
  segmentActive: { backgroundColor: colors.blue, borderColor: colors.blue },
  segmentText: { fontSize: 13, fontWeight: '700', color: colors.muted },
  hint: { fontSize: 12, color: colors.muted, marginTop: space.sm, lineHeight: 17 },
});
