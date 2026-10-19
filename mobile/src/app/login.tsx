import { Ionicons } from '@expo/vector-icons';
import { useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Button, ErrorBanner, Field } from '@/components/ui';
import { errorMessage, useAuth } from '@/lib/auth';
import { colors, radius, space } from '@/lib/theme';

export default function LoginScreen() {
  const { serverUrl, signIn } = useAuth();
  const [server, setServer] = useState(serverUrl);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [register, setRegister] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setError(null);
    if (!server.trim() || !username.trim() || !password) {
      setError('Fill in the server address, username and password.');
      return;
    }
    setBusy(true);
    try {
      await signIn(server, username, password, register);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
          <View style={styles.hero}>
            <View style={styles.logo}>
              <Ionicons name="scan-outline" size={40} color={colors.white} />
            </View>
            <Text style={styles.brand}>PanoStitch</Text>
            <Text style={styles.tagline}>Capture. Stitch. Zoom into every detail.</Text>
          </View>

          <View style={styles.panel}>
            <View style={styles.segment}>
              {(['Sign in', 'Create account'] as const).map((label, i) => {
                const active = register === (i === 1);
                return (
                  <Text
                    key={label}
                    onPress={() => setRegister(i === 1)}
                    style={[styles.segmentItem, active && styles.segmentActive]}
                    accessibilityRole="tab"
                    accessibilityState={{ selected: active }}>
                    {label}
                  </Text>
                );
              })}
            </View>

            <Field
              label="Server address"
              value={server}
              onChangeText={setServer}
              placeholder="192.168.1.20:8000"
              keyboardType="url"
            />
            <Field label="Username" value={username} onChangeText={setUsername} placeholder="your-name" textContentType="username" />
            <Field
              label="Password"
              value={password}
              onChangeText={setPassword}
              placeholder={register ? 'At least 8 characters' : 'Password'}
              secureTextEntry
              textContentType={register ? 'newPassword' : 'password'}
              onSubmitEditing={submit}
            />
            <ErrorBanner message={error} />
            <Button title={register ? 'Create account' : 'Sign in'} onPress={submit} loading={busy} icon="arrow-forward" />
            <Text style={styles.hint}>
              Run the server on your computer with{'\n'}
              <Text style={styles.code}>python -m stitcher.server --host 0.0.0.0</Text>
              {'\n'}and enter its Wi-Fi address and port above.
            </Text>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.blue },
  scroll: { flexGrow: 1, justifyContent: 'flex-end' },
  hero: { alignItems: 'center', paddingVertical: 40, gap: space.sm },
  logo: {
    width: 76,
    height: 76,
    borderRadius: 22,
    backgroundColor: colors.blueDark,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.35)',
  },
  brand: { fontSize: 30, fontWeight: '800', color: colors.white },
  tagline: { fontSize: 15, color: 'rgba(255,255,255,0.85)' },
  panel: {
    backgroundColor: colors.bg,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    padding: space.xl,
    gap: space.lg,
  },
  segment: { flexDirection: 'row', backgroundColor: colors.blueSoft, borderRadius: radius.md, padding: 4 },
  segmentItem: {
    flex: 1,
    textAlign: 'center',
    paddingVertical: 10,
    borderRadius: radius.sm,
    fontWeight: '700',
    color: colors.blueDark,
    overflow: 'hidden',
  },
  segmentActive: { backgroundColor: colors.blue, color: colors.white },
  hint: { fontSize: 12, color: colors.muted, textAlign: 'center', lineHeight: 18 },
  code: { fontFamily: Platform.select({ ios: 'Menlo', default: 'monospace' }), color: colors.blueDark },
});
