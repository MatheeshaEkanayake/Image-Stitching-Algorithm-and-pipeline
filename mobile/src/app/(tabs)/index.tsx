import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { FlatList, RefreshControl, StyleSheet, View } from 'react-native';

import { CaptureCard } from '@/components/cards';
import { Button, EmptyState, ErrorBanner } from '@/components/ui';
import { errorMessage, useAuth } from '@/lib/auth';
import { colors, space } from '@/lib/theme';
import type { Capture } from '@/lib/types';

/** The signed-in user's own captures (private library). */
export default function LibraryScreen() {
  const { api, handleUnauthorized } = useAuth();
  const [captures, setCaptures] = useState<Capture[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setCaptures(await api.library());
      setError(null);
    } catch (e) {
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }, [api, handleUnauthorized]);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  return (
    <FlatList
      style={{ backgroundColor: colors.bg }}
      contentContainerStyle={styles.list}
      data={captures}
      keyExtractor={(c) => c.id}
      renderItem={({ item }) => <CaptureCard capture={item} />}
      refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor={colors.blue} colors={[colors.blue]} />}
      ListHeaderComponent={<ErrorBanner message={error} onRetry={load} />}
      ListEmptyComponent={
        loading ? undefined : (
          <View>
            <EmptyState
              icon="aperture-outline"
              title="No panoramas yet"
              body="Capture a series of overlapping photos and the server stitches them into one high-detail image."
            />
            <Button title="Start capturing" icon="camera" onPress={() => router.navigate('/capture')} />
          </View>
        )
      }
    />
  );
}

const styles = StyleSheet.create({
  list: { padding: space.lg, gap: space.lg, flexGrow: 1 },
});
