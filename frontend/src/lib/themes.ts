export type ThemeKey = "emerald" | "blue" | "crimson" | "purple";

export interface TableTheme {
  id: ThemeKey;
  name: string;
  swatchColor: string;
  mainBg: string;
  feltBg: string;
  feltBorder: string;
  accentBorder: string;
  accentBg: string;
  accentText: string;
  accentShadow: string;
  cardRing: string;
  btnBg: string;
  btnHoverBg: string;
  btnText: string;
  jumpInColor: string;
}

export const TABLE_THEMES: Record<ThemeKey, TableTheme> = {
  emerald: {
    id: "emerald",
    name: "Classic Emerald",
    swatchColor: "#22c55e",
    mainBg:
      "radial-gradient(circle at center, #1b6338 0%, #0d381e 55%, #051a0d 100%)",
    feltBg:
      "radial-gradient(circle at top, rgba(74, 222, 128, 0.28), rgba(13, 56, 30, 0.95) 70%)",
    feltBorder: "rgba(74, 222, 128, 0.25)",
    accentBorder: "#4ade80",
    accentBg: "rgba(74, 222, 128, 0.12)",
    accentText: "#4ade80",
    accentShadow: "0 0 30px rgba(74, 222, 128, 0.25)",
    cardRing: "#4ade80",
    btnBg: "#4ade80",
    btnHoverBg: "#22c55e",
    btnText: "#052e16",
    jumpInColor: "#ef4444",
  },
  blue: {
    id: "blue",
    name: "Midnight Blue",
    swatchColor: "#38bdf8",
    mainBg:
      "radial-gradient(circle at center, #1d4ed8 0%, #1e3a8a 55%, #0f172a 100%)",
    feltBg:
      "radial-gradient(circle at top, rgba(56, 189, 248, 0.16), rgba(30, 58, 138, 0.95) 70%)",
    feltBorder: "rgba(56, 189, 248, 0.25)",
    accentBorder: "#38bdf8",
    accentBg: "rgba(56, 189, 248, 0.12)",
    accentText: "#38bdf8",
    accentShadow: "0 0 30px rgba(56, 189, 248, 0.3)",
    cardRing: "#38bdf8",
    btnBg: "#38bdf8",
    btnHoverBg: "#0284c7",
    btnText: "#082f49",
    jumpInColor: "#ef4444",
  },
  crimson: {
    id: "crimson",
    name: "Crimson Velvet",
    swatchColor: "#f43f5e",
    mainBg:
      "radial-gradient(circle at center, #881337 0%, #580a20 55%, #1c0209 100%)",
    feltBg:
      "radial-gradient(circle at top, rgba(244, 63, 94, 0.16), rgba(80, 14, 35, 0.95) 70%)",
    feltBorder: "rgba(251, 113, 133, 0.25)",
    accentBorder: "#fb7185",
    accentBg: "rgba(251, 113, 133, 0.12)",
    accentText: "#fb7185",
    accentShadow: "0 0 30px rgba(251, 113, 133, 0.3)",
    cardRing: "#fb7185",
    btnBg: "#fb7185",
    btnHoverBg: "#e11d48",
    btnText: "#4c0519",
    jumpInColor: "#facc15",
  },
  purple: {
    id: "purple",
    name: "Royal Obsidian",
    swatchColor: "#c084fc",
    mainBg:
      "radial-gradient(circle at center, #4c1d95 0%, #3b0764 55%, #120224 100%)",
    feltBg:
      "radial-gradient(circle at top, rgba(168, 85, 247, 0.16), rgba(58, 20, 120, 0.95) 70%)",
    feltBorder: "rgba(192, 132, 252, 0.25)",
    accentBorder: "#c084fc",
    accentBg: "rgba(192, 132, 252, 0.12)",
    accentText: "#c084fc",
    accentShadow: "0 0 30px rgba(192, 132, 252, 0.3)",
    cardRing: "#c084fc",
    btnBg: "#c084fc",
    btnHoverBg: "#9333ea",
    btnText: "#2e1065",
    jumpInColor: "#facc15",
  },
};
