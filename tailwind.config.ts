import type { Config } from "tailwindcss";
const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}", "./lib/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: { boxShadow: { glow: "0 0 40px rgba(34,197,94,0.18)" } } },
  plugins: []
};
export default config;
