import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 4000,
    proxy: {
      "/v1": {
        target: "http://localhost:8000",
        changeOrigin: true,
        cookieDomainRewrite: "localhost",   // rewrite cookie domain for dev
      },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
      "/ready":  { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
