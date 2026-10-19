/** Shapes returned by the stitching server (see stitcher/server/routes_*.py). */

export type UserRef = { id: number; username: string };

export type CaptureState = 'capturing' | 'queued' | 'running' | 'done' | 'failed' | 'unknown';

export type Capture = {
  id: string;
  title: string;
  owner: UserRef;
  created: number;
  is_public: boolean;
  state: CaptureState;
  photo_count: number;
  width: number | null;
  height: number | null;
  error: string | null;
};

export type Post = {
  id: number;
  body: string;
  created: number;
  author: UserRef;
  capture: Capture | null;
  like_count: number;
  comment_count: number;
  liked: boolean;
};

export type Comment = { id: number; body: string; created: number; author: UserRef };

export type Profile = {
  user: UserRef & { joined: number };
  captures: Capture[];
  posts: Post[];
};

export type Session = { token: string; expires: number; user: UserRef };
