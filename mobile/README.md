# PanoStitch mobile app

React Native (Expo SDK 57, Expo Router) client for the stitching server in `../stitcher/server`.

- **Capture**: take overlapping photos left to right; each one uploads in the background the
  moment it is taken (with retries). A ghost of the previous photo's right edge helps you line up
  the next shot. Tap **Stitch** when all photos are uploaded.
- **Library**: your own captures (private by default) with live status while the server stitches.
- **Zoom viewer**: deep-zoom tiles (only the visible tiles are downloaded) and **Full detail**,
  which fetches native-resolution pixels of the current view.
- **Community**: posts with an optional panorama, likes, comments, and user profiles showing
  each author's shared panoramas. Attaching a panorama to a post shares it.

## Run it

1. Start the server on a computer on the same Wi-Fi as the phone (from the repository root):

   ```bash
   pip install -e ".[server]"
   python -m stitcher.server --host 0.0.0.0 --port 8000
   ```

2. Start the app:

   ```bash
   cd mobile
   npm install
   npx expo start
   ```

   Scan the QR code with **Expo Go** (Android/iOS). All native modules used (camera, secure
   store, WebView) are included in Expo Go, so no custom build is needed for development.

3. On the sign-in screen enter the computer's LAN address, e.g. `192.168.1.20:8000`, then create
   an account.

## Checks

```bash
npx tsc --noEmit      # type check
npx expo lint         # lint
npx expo-doctor       # dependency/config health
```

## Notes and limits

- The server speaks plain HTTP. That is fine on a home network for development, but put it
  behind HTTPS before using it over the internet; release builds on Android and iOS block
  plain HTTP by default.
- The session token is kept in the device's secure storage. The viewer receives it in the URL
  fragment (never sent to the server) and sends it as an `Authorization` header.
- Photos are uploaded at full camera quality (`quality: 1`) to keep as much detail as possible;
  on mobile data this is several MB per photo.
- The app icon and splash are still the Expo defaults.
