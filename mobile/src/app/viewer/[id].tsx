import { useLocalSearchParams } from 'expo-router';
import { useState } from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import { WebView } from 'react-native-webview';

import { ErrorBanner } from '@/components/ui';
import { useAuth } from '@/lib/auth';
import { colors, space } from '@/lib/theme';

/**
 * Deep-zoom viewer. The server's viewer page (OpenSeadragon) loads only the tiles on screen,
 * so even very large panoramas stay smooth; "Full detail" fetches native-resolution pixels.
 */
export default function ViewerScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { api } = useAuth();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  return (
    <View style={styles.root}>
      <WebView
        source={{ uri: api.viewerUrl(id) }}
        style={styles.root}
        originWhitelist={['*']}
        onLoadEnd={() => setLoading(false)}
        onError={(e) => setError(e.nativeEvent.description || 'Could not open the viewer.')}
        onMessage={(event) => {
          try {
            const message = JSON.parse(event.nativeEvent.data) as { type: string; message?: string };
            if (message.type === 'error') setError(message.message ?? 'Viewer error');
            if (message.type === 'ready') setError(null);
          } catch {
            // ignore non-JSON messages
          }
        }}
        setSupportMultipleWindows={false}
        allowsBackForwardNavigationGestures={false}
      />
      {loading ? (
        <View style={styles.overlay} pointerEvents="none">
          <ActivityIndicator size="large" color={colors.blue} />
        </View>
      ) : null}
      {error ? (
        <View style={styles.error}>
          <ErrorBanner message={error} />
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.ink },
  overlay: { ...StyleSheet.absoluteFill, alignItems: 'center', justifyContent: 'center' },
  error: { position: 'absolute', left: space.lg, right: space.lg, top: space.lg },
});
