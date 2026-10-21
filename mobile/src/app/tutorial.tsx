import { Ionicons } from '@expo/vector-icons';
import type { ComponentProps } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { Card } from '@/components/ui';
import { colors, space } from '@/lib/theme';

type IconName = ComponentProps<typeof Ionicons>['name'];

const STEPS: { icon: IconName; title: string; body: string }[] = [
  {
    icon: 'scan-outline',
    title: 'Pick your subject',
    body: 'Start anywhere and cover the whole subject in rows, columns or a snake, in any order. Stand still and turn the phone instead of walking, so the perspective stays the same.',
  },
  {
    icon: 'layers-outline',
    title: 'Overlap by a third',
    body: 'Every photo should share about a third of its view with its neighbour. The overlap guide shows the edge of your previous photo: line it up with the same part of the scene, then shoot.',
  },
  {
    icon: 'add-outline',
    title: 'Keep the cross level',
    body: 'The white cross in the middle is fixed. The yellow cross moves as you tilt the phone, sideways and up or down. When it sits on the white one it turns green and the phone is level. For flat things like tables and boards, hold the phone parallel to the surface.',
  },
  {
    icon: 'options-outline',
    title: 'Tune the overlays',
    body: 'Tap the settings button on the camera to switch the grid, ruler, level cross and overlap guide on or off, make the guide more or less see-through, and choose which side of the frame it appears on.',
  },
  {
    icon: 'cloud-upload-outline',
    title: 'Photos upload as you shoot',
    body: 'Each photo is sent to the server in the background. Wait for every thumbnail to show a green tick, then tap Stitch.',
  },
  {
    icon: 'search-outline',
    title: 'Zoom into the result',
    body: 'Find the finished panorama in your Library. Open the zoom viewer to explore it at full detail, or share it with the community.',
  },
];

export default function TutorialScreen() {
  return (
    <ScrollView style={{ backgroundColor: colors.bg }} contentContainerStyle={styles.content}>
      <Text style={styles.intro}>Six quick steps to a sharp, seamless panorama.</Text>
      {STEPS.map((step, i) => (
        <Card key={step.title} style={styles.step}>
          <View style={styles.badge}>
            <Ionicons name={step.icon} size={22} color={colors.blueDark} />
          </View>
          <View style={{ flex: 1, gap: 4 }}>
            <Text style={styles.stepNumber}>STEP {i + 1}</Text>
            <Text style={styles.title}>{step.title}</Text>
            <Text style={styles.body}>{step.body}</Text>
          </View>
        </Card>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  content: { padding: space.lg, gap: space.md },
  intro: { fontSize: 15, color: colors.muted, marginBottom: space.xs },
  step: { flexDirection: 'row', gap: space.md, alignItems: 'flex-start' },
  badge: { width: 44, height: 44, borderRadius: 14, backgroundColor: colors.blueSoft, alignItems: 'center', justifyContent: 'center' },
  stepNumber: { fontSize: 11, fontWeight: '800', letterSpacing: 1, color: colors.blue },
  title: { fontSize: 17, fontWeight: '800', color: colors.text },
  body: { fontSize: 14, lineHeight: 20, color: colors.muted },
});
