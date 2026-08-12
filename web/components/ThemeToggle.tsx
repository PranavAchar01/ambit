"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark" | "system";

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    const stored = window.localStorage.getItem("gridsight-theme") as Theme | null;
    if (stored) setTheme(stored);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    window.localStorage.setItem("gridsight-theme", theme);
  }, [theme]);

  const next: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
  const label: Record<Theme, string> = { system: "System", light: "Light", dark: "Dark" };

  return (
    <button
      type="button"
      onClick={() => setTheme(next[theme])}
      className="rounded-md px-2.5 py-1.5 text-xs font-medium"
      style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
      aria-label={`Theme: ${label[theme]}. Activate to switch.`}
    >
      {label[theme]}
    </button>
  );
}
