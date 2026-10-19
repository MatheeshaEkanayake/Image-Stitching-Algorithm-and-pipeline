import { Stack, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from 'react-native';

import { Avatar, CaptureCard, PostCard } from '@/components/cards';
import { Card, EmptyState, ErrorBanner } from '@/components/ui';
import { errorMessage, useAuth } from '@/lib/auth';
import { colors, space } from '@/lib/theme';
import type { Post, Profile } from '@/lib/types';

/** A user's shared panoramas (with them as the author) and their posts. */
export default function ProfileScreen() {
  const { username } = useLocalSearchParams<{ username: string }>();
  const { api, handleUnauthorized } = useAuth();
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setProfile(await api.profile(username));
      setError(null);
    } catch (e) {
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    }
  }, [api, username, handleUnauthorized]);

  useEffect(() => {
    let active = true;
    api.profile(username).then(
      (p) => active && setProfile(p),
      (e) => active && !handleUnauthorized(e) && setError(errorMessage(e)),
    );
    return () => {
      active = false;
    };
  }, [api, username, handleUnauthorized]);

  async function toggleLike(post: Post) {
    try {
      const updated = await api.like(post.id, !post.liked);
      setProfile((p) => (p ? { ...p, posts: p.posts.map((x) => (x.id === updated.id ? updated : x)) } : p));
    } catch (e) {
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    }
  }

  if (!profile) {
    return (
      <View style={styles.center}>
        {error ? <ErrorBanner message={error} onRetry={load} /> : <ActivityIndicator size="large" color={colors.blue} />}
      </View>
    );
  }

  return (
    <ScrollView style={{ backgroundColor: colors.bg }} contentContainerStyle={styles.content}>
      <Stack.Screen options={{ title: profile.user.username }} />
      <Card style={styles.header}>
        <Avatar name={profile.user.username} size={64} />
        <Text style={styles.name}>{profile.user.username}</Text>
        <Text style={styles.meta}>
          Member since {new Date(profile.user.joined * 1000).toLocaleDateString()} · {profile.captures.length} shared
          panoramas · {profile.posts.length} posts
        </Text>
      </Card>

      <Text style={styles.section}>Shared panoramas</Text>
      {profile.captures.length ? (
        profile.captures.map((c) => <CaptureCard key={c.id} capture={c} />)
      ) : (
        <EmptyState icon="images-outline" title="No shared panoramas" />
      )}

      <Text style={styles.section}>Posts</Text>
      {profile.posts.length ? (
        profile.posts.map((p) => <PostCard key={p.id} post={p} onToggleLike={toggleLike} />)
      ) : (
        <EmptyState icon="chatbubble-outline" title="No posts yet" />
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: space.xl, backgroundColor: colors.bg },
  content: { padding: space.lg, gap: space.md },
  header: { alignItems: 'center', gap: space.sm, borderTopWidth: 4, borderTopColor: colors.blue },
  name: { fontSize: 22, fontWeight: '800', color: colors.text },
  meta: { fontSize: 13, color: colors.muted, textAlign: 'center' },
  section: { fontSize: 16, fontWeight: '800', color: colors.blueDark, marginTop: space.sm },
});
