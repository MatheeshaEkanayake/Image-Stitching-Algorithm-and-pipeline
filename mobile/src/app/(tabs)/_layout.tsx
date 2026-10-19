import { Ionicons } from '@expo/vector-icons';
import { Tabs } from 'expo-router/js-tabs';
import type { ComponentProps } from 'react';
import type { ColorValue } from 'react-native';

import { colors } from '@/lib/theme';

type IconName = ComponentProps<typeof Ionicons>['name'];

const icon = (active: IconName, inactive: IconName) =>
  function TabBarIcon({ focused, color, size }: { focused: boolean; color: ColorValue; size: number }) {
    return <Ionicons name={focused ? active : inactive} color={typeof color === 'string' ? color : colors.blue} size={size} />;
  };

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerStyle: { backgroundColor: colors.blue },
        headerTintColor: colors.white,
        headerTitleStyle: { fontWeight: '700' },
        tabBarActiveTintColor: colors.blue,
        tabBarInactiveTintColor: colors.muted,
        tabBarStyle: { borderTopColor: colors.blueBorder, borderTopWidth: 1 },
        tabBarLabelStyle: { fontWeight: '600' },
        sceneStyle: { backgroundColor: colors.bg },
      }}>
      <Tabs.Screen name="index" options={{ title: 'Library', tabBarIcon: icon('images', 'images-outline') }} />
      <Tabs.Screen
        name="capture"
        options={{ title: 'Capture', headerShown: false, tabBarIcon: icon('camera', 'camera-outline') }}
      />
      <Tabs.Screen name="community" options={{ title: 'Community', tabBarIcon: icon('people', 'people-outline') }} />
      <Tabs.Screen name="me" options={{ title: 'Me', tabBarIcon: icon('person-circle', 'person-circle-outline') }} />
    </Tabs>
  );
}
