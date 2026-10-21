import { router } from 'expo-router';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { Avatar } from '@/components/cards';
import { Button, Card } from '@/components/ui';
import { useAuth } from '@/lib/auth';
import { colors, space } from '@/lib/theme';

export default function MeScreen() {
  const { user, serverUrl, signOut } = useAuth();
  if (!user) return null;
  return (
    <ScrollView style={{ backgroundColor: colors.bg }} contentContainerStyle={styles.content}>
      <Card style={styles.header}>
        <Avatar name={user.username} size={80} />
        <Text style={styles.name}>{user.username}</Text>
        <Text style={styles.server}>Connected to {serverUrl}</Text>
      </Card>

      <Text style={styles.heading}>Profile</Text>
      <Button
        title="My public profile"
        icon="person-outline"
        variant="secondary"
        onPress={() => router.push({ pathname: '/user/[username]', params: { username: user.username } })}
      />
      <Button title="Write a post" icon="create-outline" variant="secondary" onPress={() => router.push('/post/new')} />

      <Text style={styles.heading}>Tutorial</Text>
      <Button title="How to capture a panorama" icon="school-outline" variant="secondary" onPress={() => router.push('/tutorial')} />

      <View style={{ height: space.md }} />
      <Button title="Sign out" icon="log-out-outline" variant="danger" onPress={signOut} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  content: { padding: space.lg, gap: space.md },
  header: { alignItems: 'center', gap: space.sm, paddingVertical: space.xl },
  name: { fontSize: 24, fontWeight: '800', color: colors.text },
  server: { fontSize: 13, color: colors.muted },
  heading: { fontSize: 12, fontWeight: '800', letterSpacing: 1, color: colors.muted, textTransform: 'uppercase', marginTop: space.sm },
});
