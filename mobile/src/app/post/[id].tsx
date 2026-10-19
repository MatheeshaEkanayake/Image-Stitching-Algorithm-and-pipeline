import { Ionicons } from '@expo/vector-icons';
import { router, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Avatar, PostCard } from '@/components/cards';
import { ErrorBanner } from '@/components/ui';
import { errorMessage, useAuth } from '@/lib/auth';
import { timeAgo } from '@/lib/format';
import { colors, radius, space } from '@/lib/theme';
import type { Comment, Post } from '@/lib/types';

export default function PostScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const postId = Number(id);
  const { api, user, handleUnauthorized } = useAuth();
  const [post, setPost] = useState<Post | null>(null);
  const [comments, setComments] = useState<Comment[]>([]);
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fail = useCallback(
    (e: unknown) => {
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    },
    [handleUnauthorized],
  );

  const load = useCallback(async () => {
    try {
      const [p, c] = await Promise.all([api.post(postId), api.comments(postId)]);
      setPost(p);
      setComments(c);
      setError(null);
    } catch (e) {
      fail(e);
    }
  }, [api, postId, fail]);

  useEffect(() => {
    let active = true;
    Promise.all([api.post(postId), api.comments(postId)]).then(
      ([p, c]) => {
        if (!active) return;
        setPost(p);
        setComments(c);
      },
      (e) => active && fail(e),
    );
    return () => {
      active = false;
    };
  }, [api, postId, fail]);

  async function toggleLike(current: Post) {
    try {
      setPost(await api.like(current.id, !current.liked));
    } catch (e) {
      fail(e);
    }
  }

  async function send() {
    const text = draft.trim();
    if (!text || !post) return;
    setSending(true);
    try {
      const comment = await api.addComment(post.id, text);
      setComments((all) => [...all, comment]);
      setPost({ ...post, comment_count: post.comment_count + 1 });
      setDraft('');
    } catch (e) {
      fail(e);
    } finally {
      setSending(false);
    }
  }

  function confirmDeletePost() {
    if (!post) return;
    Alert.alert('Delete this post?', 'Comments and likes are removed too.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            await api.deletePost(post.id);
            router.back();
          } catch (e) {
            fail(e);
          }
        },
      },
    ]);
  }

  async function deleteComment(comment: Comment) {
    try {
      await api.deleteComment(comment.id);
      setComments((all) => all.filter((c) => c.id !== comment.id));
      if (post) setPost({ ...post, comment_count: Math.max(0, post.comment_count - 1) });
    } catch (e) {
      fail(e);
    }
  }

  if (!post) {
    return (
      <View style={styles.center}>
        {error ? <ErrorBanner message={error} onRetry={load} /> : <ActivityIndicator size="large" color={colors.blue} />}
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={{ flex: 1, backgroundColor: colors.bg }}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={90}>
      <FlatList
        contentContainerStyle={styles.list}
        data={comments}
        keyExtractor={(c) => String(c.id)}
        ListHeaderComponent={
          <View style={{ gap: space.md }}>
            <ErrorBanner message={error} />
            <PostCard post={post} onToggleLike={toggleLike} />
            {post.author.id === user?.id ? (
              <Text style={styles.deletePost} onPress={confirmDeletePost}>
                Delete post
              </Text>
            ) : null}
            <Text style={styles.section}>Comments ({post.comment_count})</Text>
          </View>
        }
        renderItem={({ item }) => (
          <View style={styles.comment}>
            <Avatar name={item.author.username} size={30} />
            <View style={{ flex: 1, gap: 2 }}>
              <Text style={styles.commentAuthor}>
                {item.author.username} <Text style={styles.commentTime}>· {timeAgo(item.created)}</Text>
              </Text>
              <Text style={styles.commentBody}>{item.body}</Text>
            </View>
            {item.author.id === user?.id ? (
              <Pressable onPress={() => deleteComment(item)} hitSlop={10} accessibilityLabel="Delete comment">
                <Ionicons name="trash-outline" size={16} color={colors.muted} />
              </Pressable>
            ) : null}
          </View>
        )}
        ListEmptyComponent={<Text style={styles.noComments}>No comments yet. Start the discussion.</Text>}
      />
      <SafeAreaView edges={['bottom']} style={styles.composer}>
        <TextInput
          value={draft}
          onChangeText={setDraft}
          placeholder="Write a comment…"
          placeholderTextColor={colors.muted}
          style={styles.input}
          multiline
          maxLength={2000}
        />
        <Pressable
          onPress={send}
          disabled={!draft.trim() || sending}
          style={[styles.send, (!draft.trim() || sending) && { opacity: 0.5 }]}
          accessibilityLabel="Send comment">
          {sending ? <ActivityIndicator color={colors.white} /> : <Ionicons name="send" size={18} color={colors.white} />}
        </Pressable>
      </SafeAreaView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: space.xl, backgroundColor: colors.bg },
  list: { padding: space.lg, gap: space.md },
  deletePost: { color: colors.danger, fontWeight: '600', alignSelf: 'flex-end' },
  section: { fontSize: 15, fontWeight: '800', color: colors.blueDark, marginTop: space.sm },
  comment: {
    flexDirection: 'row',
    gap: space.sm,
    backgroundColor: colors.card,
    padding: space.md,
    borderRadius: radius.md,
    borderLeftWidth: 3,
    borderLeftColor: colors.blue,
  },
  commentAuthor: { fontWeight: '700', color: colors.text },
  commentTime: { fontWeight: '400', color: colors.muted, fontSize: 12 },
  commentBody: { color: colors.text, fontSize: 15, lineHeight: 20 },
  noComments: { color: colors.muted, textAlign: 'center', paddingVertical: space.lg },
  composer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: space.sm,
    padding: space.md,
    backgroundColor: colors.card,
    borderTopWidth: 1,
    borderTopColor: colors.blueBorder,
  },
  input: {
    flex: 1,
    maxHeight: 120,
    minHeight: 44,
    borderRadius: radius.md,
    borderWidth: 1.5,
    borderColor: colors.blueBorder,
    paddingHorizontal: space.md,
    paddingVertical: 10,
    fontSize: 15,
    color: colors.text,
  },
  send: { width: 44, height: 44, borderRadius: 22, backgroundColor: colors.blue, alignItems: 'center', justifyContent: 'center' },
});
