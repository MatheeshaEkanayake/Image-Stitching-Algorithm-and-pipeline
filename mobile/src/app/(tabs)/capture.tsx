import { Ionicons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router, useIsFocused, usePathname } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, AppState, FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { CameraGrid, CameraRuler, LevelCross } from '@/components/camera-overlays';
import { CameraSettingsSheet } from '@/components/camera-settings-sheet';
import { Button, ErrorBanner } from '@/components/ui';
import { errorMessage, useAuth } from '@/lib/auth';
import { useCameraSettings, type GuideSide } from '@/lib/camera-settings';
import { colors, radius, space } from '@/lib/theme';

type Shot = { index: number; uri: string; status: 'uploading' | 'done' | 'error'; error?: string };

const MAX_ATTEMPTS = 3;
// How much of the previous photo is shown as an alignment guide: one third, so the guide's
// edge sits on a grid line and the recommended overlap is easy to judge.
const OVERLAP_GUIDE = 1 / 3;

export default function CaptureScreen() {
  const { api, handleUnauthorized } = useAuth();
  const tabFocused = useIsFocused();
  const pathname = usePathname();
  // The tab stays "focused" while the panorama page is pushed on top of it, so also look at the
  // route: the camera is only live (and mounted) when this screen is really the one on show.
  const onScreen = tabFocused && pathname === '/capture';
  const [appActive, setAppActive] = useState(AppState.currentState === 'active');
  const cameraLive = onScreen && appActive;

  const [permission, requestPermission] = useCameraPermissions();
  const camera = useRef<CameraView>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraKey, setCameraKey] = useState(0);
  const [shots, setShots] = useState<Shot[]>([]);
  const [taking, setTaking] = useState(false);
  const [finishing, setFinishing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [settings, updateSettings] = useCameraSettings();
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    const sub = AppState.addEventListener('change', (state) => setAppActive(state === 'active'));
    return () => sub.remove();
  }, []);

  // Every time the camera comes back (from the panorama page, another tab, or the background)
  // mount a fresh CameraView: a camera surface that was kept alive underneath another screen
  // can come back half black, and the old "ready" flag no longer applies to the new view.
  useEffect(() => {
    if (cameraLive) {
      setCameraKey((k) => k + 1);
    } else {
      setCameraReady(false);
    }
  }, [cameraLive]);

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
        <View style={styles.permissionIcon}>
          <Ionicons name="camera-outline" size={40} color={colors.blueDark} />
        </View>
        <Text style={styles.permissionTitle}>Camera access</Text>
        <Text style={styles.permissionText}>PanoStitch needs the camera to capture your overlapping panorama photos.</Text>
        <Button title="Allow camera" icon="checkmark" onPress={requestPermission} />
      </SafeAreaView>
    );
  }

  const uploaded = shots.filter((s) => s.status === 'done').length;
  const failed = shots.filter((s) => s.status === 'error');
  const canFinish = shots.length >= 2 && uploaded === shots.length && !finishing;
  const last = shots[shots.length - 1];

  return (
    <View style={styles.dark} onLayout={(e) => setSize(e.nativeEvent.layout)}>
      {/* The camera and its overlays fill the whole screen; the controls float above them, so
          nothing depends on how a flex layout splits the space around the camera surface. */}
      {cameraLive ? (
        <CameraView
          key={cameraKey}
          ref={camera}
          style={StyleSheet.absoluteFill}
          facing="back"
          onCameraReady={() => setCameraReady(true)}
          onMountError={(e) => setError(e.message)}
        />
      ) : null}
      {settings.grid ? <CameraGrid /> : null}
      {settings.ruler ? <CameraRuler /> : null}
      {last && settings.guide && size.width > 0 ? (
        // Ghost of the previous photo's edge: line it up with the same part of the scene in the
        // new shot. Works for rows, columns and snakes.
        <View
          pointerEvents="none"
          style={[styles.guide, guideFrame(settings.guideSide, size), { opacity: settings.guideOpacity }]}>
          <Image source={{ uri: last.uri }} style={guideImage(settings.guideSide, size)} resizeMode="cover" />
          <View style={[styles.guideEdge, guideEdge(settings.guideSide)]} />
        </View>
      ) : null}
      {settings.level ? <LevelCross active={cameraLive} /> : null}

      <SafeAreaView edges={['top']} style={styles.topBar} pointerEvents="box-none">
        <View style={styles.topRow}>
          <View style={styles.counter}>
            <Ionicons name="images" size={14} color={colors.white} />
            <Text style={styles.counterText}>
              {shots.length} {shots.length === 1 ? 'photo' : 'photos'}
              {shots.length ? ` · ${uploaded} uploaded` : ''}
            </Text>
          </View>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Capture settings"
            onPress={() => setSettingsOpen(true)}
            style={styles.iconButton}>
            <Ionicons name="options-outline" size={22} color={colors.white} />
          </Pressable>
        </View>
        {shots.length === 0 ? (
          <Text style={styles.tip}>Overlap each shot with its neighbour by about a third.</Text>
        ) : null}
      </SafeAreaView>

      <View style={styles.tray}>
        <ErrorBanner message={error} />
        {failed.length ? (
          <ErrorBanner
            message={`${failed.length} photo(s) failed to upload. ${failed[0].error ?? ''}`}
            onRetry={() => failed.forEach((s) => void upload(s))}
          />
        ) : null}
        {shots.length ? (
          <FlatList
            horizontal
            data={shots}
            keyExtractor={(s) => String(s.index)}
            contentContainerStyle={{ gap: space.sm }}
            showsHorizontalScrollIndicator={false}
            renderItem={({ item }) => (
              <View style={styles.shot}>
                <Image source={{ uri: item.uri }} style={styles.shotImage} />
                <View
                  style={[
                    styles.shotStatus,
                    item.status === 'error' && { backgroundColor: colors.danger },
                    item.status === 'done' && { backgroundColor: colors.success },
                  ]}>
                  <Ionicons
                    name={item.status === 'uploading' ? 'cloud-upload-outline' : item.status === 'done' ? 'checkmark' : 'alert'}
                    size={11}
                    color={colors.ink}
                  />
                </View>
                <Text style={styles.shotIndex}>{item.index + 1}</Text>
              </View>
            )}
          />
        ) : null}
        <View style={styles.controls}>
          <Pressable onPress={discard} disabled={!shots.length} style={[styles.side, !shots.length && { opacity: 0.35 }]}>
            <View style={styles.sideIcon}>
              <Ionicons name="trash-outline" size={22} color={colors.white} />
            </View>
            <Text style={styles.sideText}>Discard</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Take photo"
            onPress={takePhoto}
            disabled={!cameraReady || taking}
            style={({ pressed }) => [styles.shutter, (pressed || taking) && { transform: [{ scale: 0.94 }] }, !cameraReady && { opacity: 0.5 }]}>
            <View style={styles.shutterInner} />
          </Pressable>
          <Pressable onPress={finish} disabled={!canFinish} style={[styles.side, !canFinish && { opacity: 0.35 }]}>
            <View style={[styles.sideIcon, canFinish && { backgroundColor: colors.blue, borderColor: colors.blue }]}>
              <Ionicons name={finishing ? 'hourglass-outline' : 'checkmark'} size={22} color={colors.white} />
            </View>
            <Text style={styles.sideText}>{finishing ? 'Starting…' : 'Stitch'}</Text>
          </Pressable>
        </View>
      </View>

      <CameraSettingsSheet
        visible={settingsOpen}
        settings={settings}
        onChange={updateSettings}
        onClose={() => setSettingsOpen(false)}
      />
    </View>
  );
}

type Size = { width: number; height: number };

/** Where the guide strip sits on the preview: the edge you are moving away from. */
function guideFrame(side: GuideSide, { width, height }: Size) {
  switch (side) {
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
function guideImage(side: GuideSide, { width, height }: Size) {
  const keep = 1 - OVERLAP_GUIDE;
  switch (side) {
    case 'right':
      return { width, height, marginLeft: -width * keep };
    case 'down':
      return { width, height, marginTop: -height * keep };
    default:
      return { width, height }; // left / up: the photo's left / top edge
  }
}

/** A bright line on the strip's inner edge so the overlap boundary is easy to see. */
function guideEdge(side: GuideSide) {
  switch (side) {
    case 'right':
      return { right: 0, top: 0, bottom: 0, width: 2 };
    case 'left':
      return { left: 0, top: 0, bottom: 0, width: 2 };
    case 'down':
      return { left: 0, right: 0, bottom: 0, height: 2 };
    default:
      return { left: 0, right: 0, top: 0, height: 2 };
  }
}

const styles = StyleSheet.create({
  dark: { flex: 1, backgroundColor: colors.ink },
  center: { alignItems: 'center', justifyContent: 'center', gap: space.lg, padding: space.xl },
  permissionIcon: { width: 84, height: 84, borderRadius: 28, backgroundColor: colors.blueSoft, alignItems: 'center', justifyContent: 'center' },
  permissionTitle: { color: colors.text, fontSize: 22, fontWeight: '800' },
  permissionText: { color: colors.muted, fontSize: 15, textAlign: 'center', lineHeight: 21 },
  guide: { position: 'absolute', overflow: 'hidden' },
  guideEdge: { position: 'absolute', backgroundColor: '#FFD60A' },
  topBar: { position: 'absolute', left: 0, right: 0, top: 0, paddingHorizontal: space.lg, paddingTop: space.sm, gap: space.sm },
  topRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  counter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: colors.glass,
    borderWidth: 1,
    borderColor: colors.glassBorder,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: radius.pill,
  },
  counterText: { color: colors.white, fontWeight: '700', fontSize: 13 },
  iconButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.glass,
    borderWidth: 1,
    borderColor: colors.glassBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tip: {
    alignSelf: 'center',
    color: colors.white,
    fontSize: 13,
    backgroundColor: colors.glass,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: radius.pill,
    overflow: 'hidden',
  },
  tray: {
    position: 'absolute',
    left: space.md,
    right: space.md,
    bottom: space.md,
    padding: space.lg,
    gap: space.md,
    borderRadius: 30,
    backgroundColor: colors.glass,
    borderWidth: 1,
    borderColor: colors.glassBorder,
  },
  shot: { width: 56, height: 56, borderRadius: 14, overflow: 'hidden', borderWidth: 2, borderColor: colors.glassBorder },
  shotImage: { width: '100%', height: '100%' },
  shotStatus: {
    position: 'absolute',
    right: 3,
    top: 3,
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: colors.warning,
    alignItems: 'center',
    justifyContent: 'center',
  },
  shotIndex: {
    position: 'absolute',
    left: 5,
    bottom: 2,
    color: colors.white,
    fontWeight: '800',
    fontSize: 12,
    textShadowColor: '#000',
    textShadowRadius: 3,
  },
  controls: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: space.sm },
  side: { width: 68, alignItems: 'center', gap: 6 },
  sideIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(255,255,255,0.12)',
    borderWidth: 1,
    borderColor: colors.glassBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sideText: { color: colors.white, fontSize: 12, fontWeight: '600' },
  shutter: {
    width: 78,
    height: 78,
    borderRadius: 39,
    borderWidth: 4,
    borderColor: colors.white,
    alignItems: 'center',
    justifyContent: 'center',
  },
  shutterInner: { width: 60, height: 60, borderRadius: 30, backgroundColor: colors.white },
});
