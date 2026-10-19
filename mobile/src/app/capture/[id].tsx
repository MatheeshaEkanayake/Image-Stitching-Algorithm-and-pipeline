import { Ionicons } from '@expo/vector-icons';
import { router, Stack, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Alert, Pressable, ScrollView, StyleSheet, Switch, Text, View } from 'react-native';

import { Thumbnail } from '@/components/cards';
import { Button, Card, ErrorBanner, StatusBadge } from '@/components/ui';
import { errorMessage, useAuth } from '@/lib/auth';
import { megapixels, timeAgo } from '@/lib/format';
import { colors, space } from '@/lib/theme';
import type { Capture } from '@/lib/types';

const POLL_MS = 2000;

export default function CaptureDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { api, user, handleUnauthorized } = useAuth();
  const [capture, setCapture] = useState<Capture | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    try {
      setCapture(await api.capture(id));
      setError(null);
    } catch (e) {
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    }
  }, [api, id, handleUnauthorized]);

  useEffect(() => {
    let active = true;
    api.capture(id).then(
      (c) => active && setCapture(c),
      (e) => active && !handleUnauthorized(e) && setError(errorMessage(e)),
    );
    return () => {
      active = false;
    };
  }, [api, id, handleUnauthorized]);

  // Poll while the server is stitching.
  const processing = capture?.state === 'queued' || capture?.state === 'running';
  useEffect(() => {
    if (!processing) return;
    const timer = setInterval(load, POLL_MS);
    return () => clearInterval(timer);
  }, [processing, load]);

  async function setPublic(value: boolean) {
    if (!capture) return;
    setSaving(true);
    try {
      setCapture(await api.updateCapture(capture.id, { public: value }));
    } catch (e) {
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    } finally {
      setSaving(false);
    }
  }

  function remove() {
    Alert.alert('Delete this panorama?', 'The photos, panorama and tiles are removed from the server.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            await api.deleteCapture(id);
            router.back();
          } catch (e) {
            if (!handleUnauthorized(e)) setError(errorMessage(e));
          }
        },
      },
    ]);
  }

  if (!capture) {
    return (
      <View style={styles.center}>
        {error ? <ErrorBanner message={error} onRetry={load} /> : <ActivityIndicator size="large" color={colors.blue} />}
      </View>
    );
  }

  const mine = capture.owner.id === user?.id;
  const done = capture.state === 'done';

  return (
    <ScrollView style={{ backgroundColor: colors.bg }} contentContainerStyle={styles.content}>
      <Stack.Screen options={{ title: capture.title || 'Panorama' }} />
      <ErrorBanner message={error} onRetry={load} />

      <Pressable
        disabled={!done}
        onPress={() => router.push({ pathname: '/viewer/[id]', params: { id: capture.id } })}
        style={styles.preview}>
        <Thumbnail capture={capture} height={210} />
        {done ? (
          <View style={styles.zoomButton}>
            <Ionicons name="search" size={16} color={colors.white} />
            <Text style={styles.zoomText}>Zoom into details</Text>
          </View>
        ) : null}
      </Pressable>

      <Card style={{ gap: space.sm }}>
        <View style={styles.row}>
          <StatusBadge state={capture.state} />
          <Text style={styles.meta}>
            by {capture.owner.username} · {timeAgo(capture.created)}
          </Text>
        </View>
        {processing ? (
          <View style={styles.row}>
            <ActivityIndicator color={colors.blue} />
            <Text style={styles.body}>
              Stitching {capture.photo_count} photos on the server. This can take a minute for full-resolution photos.
            </Text>
          </View>
        ) : null}
        {capture.state === 'failed' ? (
          <Text style={[styles.body, { color: colors.danger }]}>{capture.error ?? 'Stitching failed.'}</Text>
        ) : null}
        {done ? (
          <Text style={styles.body}>
            {capture.width} × {capture.height} px ({megapixels(capture.width, capture.height)}) from {capture.photo_count} photos
          </Text>
        ) : null}
      </Card>

      {mine && done ? (
        <Card style={styles.shareCard}>
          <View style={{ flex: 1 }}>
            <Text style={styles.shareTitle}>Share with the community</Text>
            <Text style={styles.meta}>
              {capture.is_public ? 'Anyone signed in can view and zoom this panorama.' : 'Only you can see it.'}
            </Text>
          </View>
          <Switch
            value={capture.is_public}
            onValueChange={setPublic}
            disabled={saving}
            trackColor={{ true: colors.blue, false: colors.line }}
            thumbColor={colors.white}
          />
        </Card>
      ) : null}

      {done ? (
        <Button
          title="Open zoom viewer"
          icon="expand-outline"
          onPress={() => router.push({ pathname: '/viewer/[id]', params: { id: capture.id } })}
        />
      ) : null}
      {mine && done ? (
        <Button
          title="Post to community"
          icon="megaphone-outline"
          variant="secondary"
          onPress={() => router.push({ pathname: '/post/new', params: { captureId: capture.id } })}
        />
      ) : null}
      {mine ? <Button title="Delete" icon="trash-outline" variant="danger" onPress={remove} /> : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: space.xl, backgroundColor: colors.bg },
  content: { padding: space.lg, gap: space.md },
  preview: { borderRadius: 18, overflow: 'hidden', borderWidth: 2, borderColor: colors.blue },
  zoomButton: {
    position: 'absolute',
    left: space.md,
    bottom: space.md,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: colors.blue,
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 999,
  },
  zoomText: { color: colors.white, fontWeight: '700' },
  row: { flexDirection: 'row', alignItems: 'center', gap: space.sm, flexWrap: 'wrap' },
  meta: { fontSize: 13, color: colors.muted },
  body: { flex: 1, fontSize: 15, color: colors.text, lineHeight: 21 },
  shareCard: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  shareTitle: { fontSize: 16, fontWeight: '700', color: colors.text },
});
