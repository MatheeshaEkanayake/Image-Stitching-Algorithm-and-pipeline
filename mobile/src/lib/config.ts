/**
 * Server the app talks to unless the user changes it. Set EXPO_PUBLIC_SERVER_URL (for example in
 * a `.env` file next to package.json: EXPO_PUBLIC_SERVER_URL=http://192.168.1.20:8000) so people
 * never have to type an address when signing up.
 */
export const DEFAULT_SERVER_URL: string = process.env.EXPO_PUBLIC_SERVER_URL ?? '';
