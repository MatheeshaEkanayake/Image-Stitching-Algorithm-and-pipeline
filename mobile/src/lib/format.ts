/** Human-friendly relative time for server timestamps (seconds since epoch). */
export function timeAgo(seconds: number): string {
  const diff = Math.max(0, Date.now() / 1000 - seconds);
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  if (diff < 7 * 86400) return `${Math.floor(diff / 86400)} d ago`;
  return new Date(seconds * 1000).toLocaleDateString();
}

export function megapixels(width: number | null, height: number | null): string {
  if (!width || !height) return '';
  return `${((width * height) / 1e6).toFixed(1)} MP`;
}
