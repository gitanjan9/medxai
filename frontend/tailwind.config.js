/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["\"Times New Roman\"", "Times", "serif"],
      },
      colors: {
        brand: {
          50: "#eff6ff",
          100: "#dbeafe",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          800: "#1e40af",
        },
      },
      boxShadow: {
        card: "0 1px 4px 0 rgba(15,23,42,0.06), 0 0 0 1px rgba(15,23,42,0.04)",
      },
    },
  },
  plugins: [],
};
