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
  // The address is configured once (EXPO_PUBLIC_SERVER_URL); it is only shown on request, or
  // when no address is known at all.
  const [showServer, setShowServer] = useState(false);
  const serverVisible = showServer || !server.trim();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [register, setRegister] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setError(null);
    if (!server.trim() || !username.trim() || !password) {
      setError(server.trim() ? 'Fill in your username and password.' : 'Fill in the server address, username and password.');
      if (!server.trim()) setShowServer(true);
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
            {serverVisible ? (
              <Field
                label="Server address"
                value={server}
                onChangeText={setServer}
                placeholder="192.168.1.20:8000"
                keyboardType="url"
              />
            ) : null}
            <ErrorBanner message={error} />
            <Button title={register ? 'Create account' : 'Sign in'} onPress={submit} loading={busy} icon="arrow-forward" />
            {!serverVisible ? (
              <Text style={styles.serverLink} onPress={() => setShowServer(true)} accessibilityRole="button">
                Connection settings
              </Text>
            ) : null}
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.ink },
  scroll: { flexGrow: 1, justifyContent: 'flex-end' },
  hero: { alignItems: 'center', paddingVertical: 40, gap: space.sm },
  logo: {
    width: 76,
    height: 76,
    borderRadius: 22,
    backgroundColor: colors.blue,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: colors.blue,
    shadowOpacity: 0.6,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 8 },
    elevation: 8,
  },
  brand: { fontSize: 30, fontWeight: '800', color: colors.white },
  tagline: { fontSize: 15, color: colors.muted },
  panel: {
    backgroundColor: colors.card,
    borderTopLeftRadius: 32,
    borderTopRightRadius: 32,
    borderWidth: 1,
    borderColor: colors.line,
    padding: space.xl,
    gap: space.lg,
  },
  segment: { flexDirection: 'row', backgroundColor: colors.inputBg, borderRadius: radius.pill, padding: 4 },
  segmentItem: {
    flex: 1,
    textAlign: 'center',
    paddingVertical: 10,
    borderRadius: radius.pill,
    fontWeight: '700',
    color: colors.muted,
    overflow: 'hidden',
  },
  segmentActive: { backgroundColor: colors.blue, color: colors.white },
  serverLink: { fontSize: 13, fontWeight: '600', color: colors.muted, textAlign: 'center', paddingVertical: 4 },
});
