import { Ionicons } from '@expo/vector-icons';
import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { FlatList, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { Thumbnail } from '@/components/cards';
import { Button, ErrorBanner, Field } from '@/components/ui';
import { errorMessage, useAuth } from '@/lib/auth';
import { colors, radius, space } from '@/lib/theme';
import type { Capture } from '@/lib/types';

/** Write a post; optionally attach one of your finished panoramas (this shares it). */
export default function NewPostScreen() {
  const { captureId } = useLocalSearchParams<{ captureId?: string }>();
  const { api, handleUnauthorized } = useAuth();
  const [body, setBody] = useState('');
  const [selected, setSelected] = useState<string | null>(captureId ?? null);
  const [mine, setMine] = useState<Capture[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .library()
      .then((all) => setMine(all.filter((c) => c.state === 'done')))
      .catch((e) => {
        if (!handleUnauthorized(e)) setError(errorMessage(e));
      });
  }, [api, handleUnauthorized]);

  async function publish() {
    if (!body.trim()) {
      setError('Write something to share.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const post = await api.createPost(body, selected);
      router.dismiss();
      router.push({ pathname: '/post/[id]', params: { id: String(post.id) } });
    } catch (e) {
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  const chosen = mine.find((c) => c.id === selected);

  return (
    <ScrollView style={{ backgroundColor: colors.bg }} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
      <Field
        label="Share an idea, tip or story"
        value={body}
        onChangeText={setBody}
        placeholder="What did you capture? Any tips for others?"
        multiline
        autoCapitalize="sentences"
        autoCorrect
        maxLength={5000}
      />
      <Text style={styles.label}>Attach a panorama (optional)</Text>
      {mine.length === 0 ? (
        <Text style={styles.hint}>Your finished panoramas will appear here.</Text>
      ) : (
        <FlatList
          horizontal
          data={mine}
          keyExtractor={(c) => c.id}
          contentContainerStyle={{ gap: space.sm }}
          showsHorizontalScrollIndicator={false}
          renderItem={({ item }) => {
            const active = item.id === selected;
            return (
              <Pressable
                onPress={() => setSelected(active ? null : item.id)}
                style={[styles.option, active && styles.optionActive]}
                accessibilityState={{ selected: active }}>
                <Thumbnail capture={item} height={80} />
                <Text numberOfLines={1} style={styles.optionTitle}>
                  {item.title || 'Untitled'}
                </Text>
                {active ? (
                  <View style={styles.check}>
                    <Ionicons name="checkmark" size={14} color={colors.white} />
                  </View>
                ) : null}
              </Pressable>
            );
          }}
        />
      )}
      {chosen && !chosen.is_public ? (
        <Text style={styles.hint}>Posting will make this panorama visible to the community.</Text>
      ) : null}
      <ErrorBanner message={error} />
      <Button title="Publish" icon="paper-plane-outline" onPress={publish} loading={busy} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  content: { padding: space.lg, gap: space.md },
  label: { fontSize: 13, fontWeight: '600', color: colors.muted },
  hint: { fontSize: 13, color: colors.muted },
  option: { width: 140, borderRadius: radius.md, overflow: 'hidden', borderWidth: 2, borderColor: colors.line, backgroundColor: colors.card },
  optionActive: { borderColor: colors.blue },
  optionTitle: { padding: 6, fontSize: 12, fontWeight: '600', color: colors.text },
  check: {
    position: 'absolute',
    top: 6,
    right: 6,
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: colors.blue,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
