import { Ionicons } from '@expo/vector-icons';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useRef, useState } from 'react';
import { ActivityIndicator, FlatList, Pressable, RefreshControl, StyleSheet } from 'react-native';

import { PostCard } from '@/components/cards';
import { EmptyState, ErrorBanner } from '@/components/ui';
import { errorMessage, useAuth } from '@/lib/auth';
import { colors, shadow, space } from '@/lib/theme';
import type { Post } from '@/lib/types';

export default function CommunityScreen() {
  const { api, handleUnauthorized } = useAuth();
  const [posts, setPosts] = useState<Post[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reachedEnd = useRef(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const page = await api.feed();
      reachedEnd.current = page.length < 20;
      setPosts(page);
      setError(null);
    } catch (e) {
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    } finally {
      setRefreshing(false);
    }
  }, [api, handleUnauthorized]);

  useFocusEffect(
    useCallback(() => {
      void refresh();
    }, [refresh]),
  );

  async function loadMore() {
    if (loadingMore || refreshing || reachedEnd.current || posts.length === 0) return;
    setLoadingMore(true);
    try {
      const page = await api.feed(posts[posts.length - 1].id);
      reachedEnd.current = page.length < 20;
      setPosts((current) => [...current, ...page.filter((p) => !current.some((c) => c.id === p.id))]);
    } catch (e) {
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    } finally {
      setLoadingMore(false);
    }
  }

  async function toggleLike(post: Post) {
    // optimistic update, then reconcile with the server's counts
    const flip = (p: Post): Post =>
      p.id === post.id ? { ...p, liked: !post.liked, like_count: p.like_count + (post.liked ? -1 : 1) } : p;
    setPosts((all) => all.map(flip));
    try {
      const updated = await api.like(post.id, !post.liked);
      setPosts((all) => all.map((p) => (p.id === updated.id ? updated : p)));
    } catch (e) {
      setPosts((all) => all.map((p) => (p.id === post.id ? post : p)));
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    }
  }

  return (
    <>
      <FlatList
        style={{ backgroundColor: colors.bg }}
        contentContainerStyle={styles.list}
        data={posts}
        keyExtractor={(p) => String(p.id)}
        renderItem={({ item }) => <PostCard post={item} onToggleLike={toggleLike} />}
        onEndReached={loadMore}
        onEndReachedThreshold={0.4}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refresh} tintColor={colors.blue} colors={[colors.blue]} />}
        ListHeaderComponent={<ErrorBanner message={error} onRetry={refresh} />}
        ListFooterComponent={loadingMore ? <ActivityIndicator color={colors.blue} /> : undefined}
        ListEmptyComponent={
          refreshing ? undefined : (
            <EmptyState icon="chatbubbles-outline" title="No posts yet" body="Share a panorama or an idea to start the conversation." />
          )
        }
      />
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="New post"
        onPress={() => router.push('/post/new')}
        style={({ pressed }) => [styles.fab, pressed && { opacity: 0.85 }]}>
        <Ionicons name="add" size={30} color={colors.white} />
      </Pressable>
    </>
  );
}

const styles = StyleSheet.create({
  list: { padding: space.lg, gap: space.lg, paddingBottom: 96, flexGrow: 1 },
  fab: {
    position: 'absolute',
    right: space.xl,
    bottom: space.xl,
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: colors.blue,
    alignItems: 'center',
    justifyContent: 'center',
    ...shadow,
    shadowOpacity: 0.3,
    elevation: 6,
  },
});
