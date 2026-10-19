import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { ActivityIndicator, View } from 'react-native';

import { AuthProvider, useAuth } from '@/lib/auth';
import { colors } from '@/lib/theme';

function RootStack() {
  const { ready, user } = useAuth();
  if (!ready) {
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.bg }}>
        <ActivityIndicator size="large" color={colors.blue} />
      </View>
    );
  }
  const signedIn = !!user;
  return (
    <Stack
      screenOptions={{
        headerStyle: { backgroundColor: colors.blue },
        headerTintColor: colors.white,
        headerTitleStyle: { fontWeight: '700' },
        contentStyle: { backgroundColor: colors.bg },
      }}>
      <Stack.Protected guard={signedIn}>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="capture/[id]" options={{ title: 'Panorama' }} />
        <Stack.Screen name="viewer/[id]" options={{ title: 'Explore detail', headerStyle: { backgroundColor: colors.ink } }} />
        <Stack.Screen name="post/[id]" options={{ title: 'Post' }} />
        <Stack.Screen name="post/new" options={{ title: 'New post', presentation: 'modal' }} />
        <Stack.Screen name="user/[username]" options={{ title: 'Profile' }} />
      </Stack.Protected>
      <Stack.Protected guard={!signedIn}>
        <Stack.Screen name="login" options={{ headerShown: false }} />
      </Stack.Protected>
    </Stack>
  );
}

export default function RootLayout() {
  return (
    <AuthProvider>
      <StatusBar style="light" />
      <RootStack />
    </AuthProvider>
  );
}
