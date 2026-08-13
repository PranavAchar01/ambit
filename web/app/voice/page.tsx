import { redirect } from "next/navigation";

/**
 * Voice is a dock on every page now, not a destination.
 *
 * Kept as a redirect rather than deleted: the QR/tunnel notes and at least one
 * screenshot in the repo point at /voice, and a 404 there reads as a broken
 * build rather than as a deliberate move.
 */
export default function VoicePage() {
  redirect("/");
}
