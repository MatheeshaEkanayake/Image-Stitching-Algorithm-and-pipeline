import { Ionicons } from '@expo/vector-icons';
import type { ComponentProps, ReactNode } from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
  type StyleProp,
  type TextInputProps,
  type ViewStyle,
} from 'react-native';

import { colors, radius, shadow, space } from '@/lib/theme';
import type { CaptureState } from '@/lib/types';

type ButtonProps = {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  icon?: ComponentProps<typeof Ionicons>['name'];
  loading?: boolean;
  disabled?: boolean;
  style?: StyleProp<ViewStyle>;
};

export function Button({ title, onPress, variant = 'primary', icon, loading, disabled, style }: ButtonProps) {
  const palette = {
    primary: { bg: colors.blue, fg: colors.white, border: colors.blue },
    secondary: { bg: colors.blueSoft, fg: colors.blueDark, border: colors.blueBorder },
    danger: { bg: colors.dangerSoft, fg: colors.danger, border: colors.dangerSoft },
    ghost: { bg: 'transparent', fg: colors.blue, border: 'transparent' },
  }[variant];
  const inactive = disabled || loading;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled: !!inactive, busy: !!loading }}
      onPress={onPress}
      disabled={inactive}
      style={({ pressed }) => [
        styles.button,
        { backgroundColor: palette.bg, borderColor: palette.border, opacity: inactive ? 0.55 : pressed ? 0.85 : 1 },
        style,
      ]}>
      {loading ? (
        <ActivityIndicator color={palette.fg} />
      ) : (
        <>
          {icon ? <Ionicons name={icon} size={18} color={palette.fg} /> : null}
          <Text style={[styles.buttonText, { color: palette.fg }]}>{title}</Text>
        </>
      )}
    </Pressable>
  );
}

export function Field({ label, ...props }: TextInputProps & { label: string }) {
  return (
    <View style={{ gap: space.xs }}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        placeholderTextColor={colors.muted}
        autoCapitalize="none"
        autoCorrect={false}
        {...props}
        style={[styles.input, props.multiline && { minHeight: 110, textAlignVertical: 'top' }, props.style]}
      />
    </View>
  );
}

export function Card({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

const STATE_STYLE: Record<CaptureState, { label: string; fg: string; bg: string }> = {
  capturing: { label: 'Capturing', fg: colors.blueDark, bg: colors.blueSoft },
  queued: { label: 'Queued', fg: colors.warning, bg: colors.warningSoft },
  running: { label: 'Stitching…', fg: colors.warning, bg: colors.warningSoft },
  done: { label: 'Ready', fg: colors.success, bg: colors.successSoft },
  failed: { label: 'Failed', fg: colors.danger, bg: colors.dangerSoft },
  unknown: { label: 'Unknown', fg: colors.muted, bg: colors.line },
};

export function StatusBadge({ state }: { state: CaptureState }) {
  const s = STATE_STYLE[state] ?? STATE_STYLE.unknown;
  return (
    <View style={[styles.badge, { backgroundColor: s.bg }]}>
      <Text style={[styles.badgeText, { color: s.fg }]}>{s.label}</Text>
    </View>
  );
}

export function Pill({ text, icon }: { text: string; icon?: ComponentProps<typeof Ionicons>['name'] }) {
  return (
    <View style={[styles.badge, { backgroundColor: colors.blueSoft, flexDirection: 'row', gap: 4 }]}>
      {icon ? <Ionicons name={icon} size={12} color={colors.blueDark} /> : null}
      <Text style={[styles.badgeText, { color: colors.blueDark }]}>{text}</Text>
    </View>
  );
}

export function EmptyState({ icon, title, body }: { icon: ComponentProps<typeof Ionicons>['name']; title: string; body?: string }) {
  return (
    <View style={styles.empty}>
      <View style={styles.emptyIcon}>
        <Ionicons name={icon} size={32} color={colors.blue} />
      </View>
      <Text style={styles.emptyTitle}>{title}</Text>
      {body ? <Text style={styles.emptyBody}>{body}</Text> : null}
    </View>
  );
}

export function ErrorBanner({ message, onRetry }: { message: string | null; onRetry?: () => void }) {
  if (!message) return null;
  return (
    <View style={styles.error}>
      <Ionicons name="alert-circle" size={18} color={colors.danger} />
      <Text style={styles.errorText}>{message}</Text>
      {onRetry ? (
        <Text style={styles.errorRetry} onPress={onRetry}>
          Retry
        </Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  button: {
    minHeight: 48,
    paddingHorizontal: space.lg,
    borderRadius: radius.md,
    borderWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: space.sm,
  },
  buttonText: { fontSize: 16, fontWeight: '600' },
  label: { fontSize: 13, fontWeight: '600', color: colors.muted },
  input: {
    minHeight: 48,
    borderRadius: radius.md,
    borderWidth: 1.5,
    borderColor: colors.blueBorder,
    backgroundColor: colors.white,
    paddingHorizontal: space.md,
    paddingVertical: space.sm,
    fontSize: 16,
    color: colors.text,
  },
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: space.lg,
    borderWidth: 1,
    borderColor: colors.line,
    ...shadow,
  },
  badge: { alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 4, borderRadius: radius.pill, alignItems: 'center' },
  badgeText: { fontSize: 12, fontWeight: '700' },
  empty: { alignItems: 'center', paddingVertical: 48, paddingHorizontal: space.xl, gap: space.sm },
  emptyIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: colors.blueSoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: space.sm,
  },
  emptyTitle: { fontSize: 17, fontWeight: '700', color: colors.text, textAlign: 'center' },
  emptyBody: { fontSize: 14, color: colors.muted, textAlign: 'center', lineHeight: 20 },
  error: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
    backgroundColor: colors.dangerSoft,
    borderRadius: radius.md,
    padding: space.md,
  },
  errorText: { flex: 1, color: colors.danger, fontSize: 14 },
  errorRetry: { color: colors.blue, fontWeight: '700' },
});
