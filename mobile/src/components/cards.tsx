import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';

import { useAuth } from '@/lib/auth';
import { megapixels, timeAgo } from '@/lib/format';
import { colors, radius, shadow, space } from '@/lib/theme';
import type { Capture, Post } from '@/lib/types';
import { Pill, StatusBadge } from './ui';

/** Panorama thumbnail; the server needs the session token, sent as a header. */
export function Thumbnail({ capture, height = 140 }: { capture: Capture; height?: number }) {
  const { api } = useAuth();
  if (capture.state !== 'done') {
    return (
      <View style={[styles.thumbPlaceholder, { height }]}>
        <Ionicons name={capture.state === 'failed' ? 'warning-outline' : 'images-outline'} size={28} color={colors.blue} />
      </View>
    );
  }
  return (
    <Image
      source={{ uri: api.thumbUrl(capture.id), headers: api.authHeaders() }}
      style={[styles.thumb, { height }]}
      resizeMode="cover"
      accessibilityLabel={`Panorama ${capture.title || 'untitled'}`}
    />
  );
}

export function CaptureCard({ capture }: { capture: Capture }) {
  return (
    <Pressable
      onPress={() => router.push({ pathname: '/capture/[id]', params: { id: capture.id } })}
      style={({ pressed }) => [styles.card, pressed && { opacity: 0.9 }]}>
      <Thumbnail capture={capture} />
      <View style={styles.cardBody}>
        <View style={styles.row}>
          <Text style={styles.title} numberOfLines={1}>
            {capture.title || 'Untitled panorama'}
          </Text>
          <StatusBadge state={capture.state} />
        </View>
        <View style={styles.row}>
          <Text style={styles.meta}>
            {capture.photo_count} photos · {timeAgo(capture.created)}
            {capture.width ? ` · ${megapixels(capture.width, capture.height)}` : ''}
          </Text>
          {capture.is_public ? <Pill text="Shared" icon="globe-outline" /> : <Pill text="Private" icon="lock-closed-outline" />}
        </View>
      </View>
    </Pressable>
  );
}

export function PostCard({ post, onToggleLike }: { post: Post; onToggleLike?: (post: Post) => void }) {
  return (
    <Pressable
      onPress={() => router.push({ pathname: '/post/[id]', params: { id: String(post.id) } })}
      style={({ pressed }) => [styles.card, pressed && { opacity: 0.92 }]}>
      <View style={styles.cardBody}>
        <Pressable
          onPress={() => router.push({ pathname: '/user/[username]', params: { username: post.author.username } })}
          style={styles.authorRow}
          hitSlop={8}>
          <Avatar name={post.author.username} />
          <View>
            <Text style={styles.author}>{post.author.username}</Text>
            <Text style={styles.meta}>{timeAgo(post.created)}</Text>
          </View>
        </Pressable>
        <Text style={styles.body}>{post.body}</Text>
      </View>
      {post.capture ? (
        <Pressable onPress={() => router.push({ pathname: '/viewer/[id]', params: { id: post.capture!.id } })}>
          <Thumbnail capture={post.capture} height={170} />
          <View style={styles.zoomHint}>
            <Ionicons name="search" size={14} color={colors.white} />
            <Text style={styles.zoomHintText}>Tap to zoom</Text>
          </View>
        </Pressable>
      ) : null}
      <View style={[styles.cardBody, styles.actions]}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={post.liked ? 'Unlike' : 'Like'}
          onPress={() => onToggleLike?.(post)}
          style={styles.action}
          hitSlop={8}>
          <Ionicons name={post.liked ? 'heart' : 'heart-outline'} size={20} color={post.liked ? colors.blue : colors.muted} />
          <Text style={[styles.actionText, post.liked && { color: colors.blue }]}>{post.like_count}</Text>
        </Pressable>
        <View style={styles.action}>
          <Ionicons name="chatbubble-outline" size={19} color={colors.muted} />
          <Text style={styles.actionText}>{post.comment_count}</Text>
        </View>
      </View>
    </Pressable>
  );
}

export function Avatar({ name, size = 36 }: { name: string; size?: number }) {
  return (
    <View style={[styles.avatar, { width: size, height: size, borderRadius: size / 2 }]}>
      <Text style={[styles.avatarText, { fontSize: size * 0.42 }]}>{name.slice(0, 1).toUpperCase()}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.line,
    overflow: 'hidden',
    ...shadow,
  },
  cardBody: { padding: space.md, gap: space.sm },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: space.sm },
  title: { flex: 1, fontSize: 16, fontWeight: '700', color: colors.text },
  meta: { fontSize: 12, color: colors.muted },
  thumb: { width: '100%', backgroundColor: colors.ink },
  thumbPlaceholder: { width: '100%', backgroundColor: colors.blueSoft, alignItems: 'center', justifyContent: 'center' },
  authorRow: { flexDirection: 'row', alignItems: 'center', gap: space.sm, alignSelf: 'flex-start' },
  author: { fontSize: 15, fontWeight: '700', color: colors.text },
  body: { fontSize: 15, lineHeight: 21, color: colors.text },
  zoomHint: {
    position: 'absolute',
    right: space.sm,
    bottom: space.sm,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(30,111,255,0.9)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: radius.pill,
  },
  zoomHintText: { color: colors.white, fontSize: 12, fontWeight: '700' },
  actions: { flexDirection: 'row', gap: space.xl, paddingTop: 0 },
  action: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  actionText: { fontSize: 14, color: colors.muted, fontWeight: '600' },
  avatar: { backgroundColor: colors.blue, alignItems: 'center', justifyContent: 'center' },
  avatarText: { color: colors.white, fontWeight: '800' },
});
