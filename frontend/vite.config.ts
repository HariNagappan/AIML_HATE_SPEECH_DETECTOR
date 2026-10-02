import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend URL is configured through VITE_API_BASE_URL (see .env.example).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Bind IPv4 explicitly: "localhost" can resolve to IPv6-only on some
    // Windows machines, which breaks IPv4 health checks and tooling.
    host: "127.0.0.1",
  },
  preview: {
    // 4173 falls inside a Windows reserved port range on some machines
    // (Hyper-V / WinNAT exclusions) - use a safe default instead.
    port: 8765,
    host: "127.0.0.1",
  },
});
