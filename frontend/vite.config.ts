import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5001,
    allowedHosts: true,
    proxy: {
    "/process-video": "http://0.0.0.0:8000", // match your FastAPI port
  },
  },
});
