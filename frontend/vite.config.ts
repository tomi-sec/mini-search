import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Proxies /search to the FastAPI backend during `npm run dev`, so the UI
// can just call a relative URL instead of hardcoding a host - api.py
// already sets permissive CORS too, so a built/served frontend works
// without this proxy either way.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/search": "http://localhost:8000",
    },
  },
});
