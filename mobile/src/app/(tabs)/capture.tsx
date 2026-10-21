import { Ionicons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router, useIsFocused } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import { useCallback, useEffect, useRef, useState, type ComponentProps } from 'react';
import { Alert, FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { CameraGrid, LevelBadge, useLevel } from '@/components/camera-overlays';
import { Button, ErrorBanner } from '@/components/ui';
import { errorMessage, useAuth } from '@/lib/auth';
import { colors, radius, space } from '@/lib/theme';

type Shot = { index: number; uri: string; status: 'uploading' | 'done' | 'error'; error?: string };

const MAX_ATTEMPTS = 3;
// How much of the previous photo is shown as an alignment guide: one third, so the guide's
// edge sits on a grid line and the recommended overlap is easy to judge.
const OVERLAP_GUIDE = 1 / 3;
const GRID_PREF = 'camera_grid';

/** Direction you are moving the camera next; the guide shows the previous photo's overlap edge. */
type Direction = 'right' | 'left' | 'down' | 'up' | 'off';
const DIRECTIONS: { key: Direction; icon: ComponentProps<typeof Ionicons>['name']; label: string }[] = [
  { key: 'right', icon: 'arrow-forward', label: 'Moving right' },
  { key: 'left', icon: 'arrow-back', label: 'Moving left' },
  { key: 'down', icon: 'arrow-down', label: 'Moving down' },
  { key: 'up', icon: 'arrow-up', label: 'Moving up' },
  { key: 'off', icon: 'eye-off-outline', label: 'Hide guide' },
];

export default function CaptureScreen() {
  const { api, handleUnauthorized } = useAuth();
  const focused = useIsFocused();
  const [permission, requestPermission] = useCameraPermissions();
  const camera = useRef<CameraView>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [shots, setShots] = useState<Shot[]>([]);
  const [taking, setTaking] = useState(false);
  const [finishing, setFinishing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [direction, setDirection] = useState<Direction>('right');
  const [gridOn, setGridOn] = useState(true);
  const level = useLevel(focused && gridOn);

  useEffect(() => {
    SecureStore.getItemAsync(GRID_PREF).then(
      (value) => value !== null && setGridOn(value === '1'),
      () => undefined,
    );
  }, []);

  function toggleGrid() {
    const next = !gridOn;
    setGridOn(next);
    SecureStore.setItemAsync(GRID_PREF, next ? '1' : '0').catch(() => undefined);
  }
  // One capture per session; created lazily on the first shot. A promise so that quick
  // successive shots all wait for the same capture instead of creating several.
  const captureId = useRef<Promise<string> | null>(null);
  const nextIndex = useRef(0);

  const setShot = (index: number, changes: Partial<Shot>) =>
    setShots((all) => all.map((s) => (s.index === index ? { ...s, ...changes } : s)));

  const upload = useCallback(
    async (shot: Shot) => {
      setShot(shot.index, { status: 'uploading', error: undefined });
      for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        try {
          const id = await captureId.current!;
          await api.uploadPhoto(id, shot.index, shot.uri);
          setShot(shot.index, { status: 'done' });
          return;
        } catch (e) {
          if (handleUnauthorized(e)) return;
          if (attempt === MAX_ATTEMPTS) {
            setShot(shot.index, { status: 'error', error: errorMessage(e) });
            return;
          }
          await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
        }
      }
    },
    [api, handleUnauthorized],
  );

  async function takePhoto() {
    if (!camera.current || !cameraReady || taking) return;
    setTaking(true);
    setError(null);
    try {
      if (!captureId.current) {
        const created = api.createCapture(`Panorama ${new Date().toLocaleString()}`).then((c) => c.id);
        captureId.current = created;
        created.catch(() => (captureId.current = null)); // allow a retry on the next shot
      }
      // quality 1 = least JPEG compression: the server keeps as much detail as possible
      const photo = await camera.current.takePictureAsync({ quality: 1, shutterSound: false });
      if (!photo?.uri) throw new Error('The camera did not return a photo.');
      const shot: Shot = { index: nextIndex.current++, uri: photo.uri, status: 'uploading' };
      setShots((all) => [...all, shot]);
      void upload(shot); // upload right away, in the background
    } catch (e) {
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    } finally {
      setTaking(false);
    }
  }

  function reset() {
    captureId.current = null;
    nextIndex.current = 0;
    setShots([]);
    setError(null);
  }

  async function finish() {
    if (!captureId.current) return;
    setFinishing(true);
    setError(null);
    try {
      const id = await captureId.current;
      await api.finishCapture(id);
      reset();
      router.push({ pathname: '/capture/[id]', params: { id } });
    } catch (e) {
      if (!handleUnauthorized(e)) setError(errorMessage(e));
    } finally {
      setFinishing(false);
    }
  }

  function discard() {
    Alert.alert('Discard these photos?', 'They will be removed from the server too.', [
      { text: 'Keep', style: 'cancel' },
      {
        text: 'Discard',
        style: 'destructive',
        onPress: async () => {
          const pending = captureId.current;
          reset();
          if (pending) await pending.then((id) => api.deleteCapture(id)).catch(() => undefined);
        },
      },
    ]);
  }

  if (!permission) return <View style={styles.dark} />;
  if (!permission.granted) {
    return (
      <SafeAreaView style={[styles.dark, styles.center]}>
        <Ionicons name="camera-outline" size={48} color={colors.white} />
        <Text style={styles.permissionText}>Camera access is needed to capture panorama photos.</Text>
        <Button title="Allow camera" icon="checkmark" onPress={requestPermission} />
      </SafeAreaView>
    );
  }

  const uploaded = shots.filter((s) => s.status === 'done').length;
  const failed = shots.filter((s) => s.status === 'error');
  const canFinish = shots.length >= 2 && uploaded === shots.length && !finishing;
  const last = shots[shots.length - 1];

  return (
    <View style={styles.dark}>
      <View style={styles.preview} onLayout={(e) => setSize(e.nativeEvent.layout)}>
        {focused ? (
          <CameraView
            ref={camera}
            style={StyleSheet.absoluteFill}
            facing="back"
            onCameraReady={() => setCameraReady(true)}
            onMountError={(e) => setError(e.message)}
          />
        ) : null}
        {gridOn ? <CameraGrid /> : null}
        {last && direction !== 'off' && size.width > 0 ? (
          // Ghost of the previous photo's edge on the side you are coming from: line it up with
          // the same part of the scene in the new shot. Works for rows, columns and snakes.
          <View pointerEvents="none" style={[styles.guide, guideFrame(direction, size)]}>
            <Image source={{ uri: last.uri }} style={guideImage(direction, size)} resizeMode="cover" />
          </View>
        ) : null}
        <SafeAreaView edges={['top']} style={styles.topBar}>
          <View style={styles.badges}>
            <View style={styles.counter}>
              <Ionicons name="images" size={14} color={colors.white} />
              <Text style={styles.counterText}>
                {shots.length} taken · {uploaded} uploaded
              </Text>
            </View>
            <LevelBadge level={level} />
          </View>
          <Text style={styles.tip}>
            {shots.length === 0
              ? 'Start anywhere and cover the whole subject in rows, columns or a snake, in any order. Overlap each shot with its neighbours by about a third. Tap the grid button to show the grid and level; for flat things like boards, keep the phone parallel until the badge turns green.'
              : 'Line the faint ghost up with the same part of the scene, then shoot. Pick the arrow for the way you move next.'}
          </Text>
          <View style={styles.directions}>
            <Pressable
              accessibilityRole="switch"
              accessibilityLabel="Grid and level"
              accessibilityState={{ checked: gridOn }}
              onPress={toggleGrid}
              style={[styles.directionButton, gridOn && styles.directionActive, { marginRight: space.sm }]}>
              <Ionicons name="grid-outline" size={20} color={colors.white} />
            </Pressable>
            {DIRECTIONS.map((d) => (
              <Pressable
                key={d.key}
                accessibilityRole="button"
                accessibilityLabel={d.label}
                accessibilityState={{ selected: direction === d.key }}
                onPress={() => setDirection(d.key)}
                style={[styles.directionButton, direction === d.key && styles.directionActive]}>
                <Ionicons name={d.icon} size={20} color={colors.white} />
              </Pressable>
            ))}
          </View>
        </SafeAreaView>
      </View>

      <View style={styles.tray}>
        <ErrorBanner message={error} />
        {failed.length ? (
          <ErrorBanner
            message={`${failed.length} photo(s) failed to upload. ${failed[0].error ?? ''}`}
            onRetry={() => failed.forEach((s) => void upload(s))}
          />
        ) : null}
        <FlatList
          horizontal
          data={shots}
          keyExtractor={(s) => String(s.index)}
          contentContainerStyle={{ gap: space.sm }}
          showsHorizontalScrollIndicator={false}
          renderItem={({ item }) => (
            <View style={styles.shot}>
              <Image source={{ uri: item.uri }} style={styles.shotImage} />
              <View style={[styles.shotStatus, item.status === 'error' && { backgroundColor: colors.danger }]}>
                {item.status === 'uploading' ? (
                  <Ionicons name="cloud-upload-outline" size={12} color={colors.white} />
                ) : item.status === 'done' ? (
                  <Ionicons name="checkmark" size={12} color={colors.white} />
                ) : (
                  <Ionicons name="alert" size={12} color={colors.white} />
                )}
              </View>
              <Text style={styles.shotIndex}>{item.index + 1}</Text>
            </View>
          )}
          ListEmptyComponent={<Text style={styles.trayHint}>Photos appear here and upload as you shoot.</Text>}
        />
        <View style={styles.controls}>
          <Pressable onPress={discard} disabled={!shots.length} style={[styles.side, !shots.length && { opacity: 0.4 }]}>
            <Ionicons name="trash-outline" size={24} color={colors.white} />
            <Text style={styles.sideText}>Discard</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Take photo"
            onPress={takePhoto}
            disabled={!cameraReady || taking}
            style={({ pressed }) => [styles.shutter, (pressed || taking) && { transform: [{ scale: 0.94 }] }]}>
            <View style={styles.shutterInner} />
          </Pressable>
          <Pressable onPress={finish} disabled={!canFinish} style={[styles.side, !canFinish && { opacity: 0.4 }]}>
            <Ionicons name={finishing ? 'hourglass-outline' : 'checkmark-circle'} size={26} color={colors.white} />
            <Text style={styles.sideText}>{finishing ? 'Starting…' : 'Stitch'}</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
}

type Size = { width: number; height: number };

/** Where the guide strip sits on the preview: the edge you are moving away from. */
function guideFrame(direction: Direction, { width, height }: Size) {
  switch (direction) {
    case 'right':
      return { left: 0, top: 0, bottom: 0, width: width * OVERLAP_GUIDE };
    case 'left':
      return { right: 0, top: 0, bottom: 0, width: width * OVERLAP_GUIDE };
    case 'down':
      return { left: 0, right: 0, top: 0, height: height * OVERLAP_GUIDE };
    default:
      return { left: 0, right: 0, bottom: 0, height: height * OVERLAP_GUIDE };
  }
}

/** The previous photo, shifted so only its edge nearest the new shot shows in the strip. */
function guideImage(direction: Direction, { width, height }: Size) {
  const keep = 1 - OVERLAP_GUIDE;
  switch (direction) {
    case 'right':
      return { width, height, marginLeft: -width * keep };
    case 'down':
      return { width, height, marginTop: -height * keep };
    default:
      return { width, height }; // left / up: the photo's left / top edge
  }
}

const styles = StyleSheet.create({
  dark: { flex: 1, backgroundColor: colors.ink },
  center: { alignItems: 'center', justifyContent: 'center', gap: space.lg, padding: space.xl },
  permissionText: { color: colors.white, fontSize: 16, textAlign: 'center' },
  preview: { flex: 1, overflow: 'hidden' },
  guide: { position: 'absolute', overflow: 'hidden', opacity: 0.38 },
  directions: { flexDirection: 'row', gap: space.sm, alignSelf: 'flex-start' },
  badges: { flexDirection: 'row', gap: space.sm, flexWrap: 'wrap', alignItems: 'center' },
  directionButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(11,18,32,0.6)',
    borderWidth: 1.5,
    borderColor: colors.blueBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  directionActive: { backgroundColor: colors.blue, borderColor: colors.white },
  topBar: { position: 'absolute', left: 0, right: 0, top: 0, padding: space.md, gap: space.sm },
  counter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    alignSelf: 'flex-start',
    backgroundColor: colors.blue,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: radius.pill,
  },
  counterText: { color: colors.white, fontWeight: '700', fontSize: 13 },
  tip: {
    color: colors.white,
    fontSize: 13,
    backgroundColor: 'rgba(11,18,32,0.6)',
    padding: space.sm,
    borderRadius: radius.sm,
    overflow: 'hidden',
  },
  tray: { padding: space.md, gap: space.md, borderTopWidth: 2, borderTopColor: colors.blue },
  trayHint: { color: colors.muted, fontSize: 13, paddingVertical: 20 },
  shot: { width: 64, height: 64, borderRadius: radius.sm, overflow: 'hidden', borderWidth: 2, borderColor: colors.blue },
  shotImage: { width: '100%', height: '100%' },
  shotStatus: {
    position: 'absolute',
    right: 2,
    top: 2,
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: colors.blue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  shotIndex: {
    position: 'absolute',
    left: 4,
    bottom: 2,
    color: colors.white,
    fontWeight: '800',
    fontSize: 12,
    textShadowColor: '#000',
    textShadowRadius: 3,
  },
  controls: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: space.lg },
  side: { width: 72, alignItems: 'center', gap: 4 },
  sideText: { color: colors.white, fontSize: 12, fontWeight: '600' },
  shutter: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 4,
    borderColor: colors.blue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  shutterInner: { width: 58, height: 58, borderRadius: 29, backgroundColor: colors.white },
});
