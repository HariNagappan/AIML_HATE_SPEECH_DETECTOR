/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
    // Test files only contain fixture text; keep them out of the CSS scan.
    "!./src/**/*.test.ts",
    "!./src/**/*.test.tsx",
  ],
  theme: {
    extend: {
      colors: {
        base: "#0a0c0f",
        panel: "#101419",
        raised: "#161b22",
        edge: "#232b34",
        "edge-strong": "#33404d",
        ink: "#e8ecf0",
        "ink-soft": "#a7b1bc",
        "ink-mute": "#75808c",
        accent: {
          DEFAULT: "#5ea6f8",
          soft: "#1d3a57",
        },
        tone: {
          normal: "#3fb27f",
          offensive: "#d9a441",
          hate: "#e0524d",
          neutral: "#8b95a1",
          info: "#5ea6f8",
        },
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "-apple-system", "sans-serif"],
        mono: [
          "Cascadia Mono",
          "Consolas",
          "SFMono-Regular",
          "ui-monospace",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
