import { writable, type Writable } from "svelte/store";

// A store kept in localStorage. Stored fields of the wrong type are ignored, so
// adding a field or a corrupt entry falls back to the defaults
export function persisted<T extends object>(key: string, defaults: T): Writable<T> {
  const store = writable<T>({ ...defaults, ...readStored(key, defaults) });

  store.subscribe((value) => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Storage is blocked or full; the setting just won't survive a reload
    }
  });

  return store;
}

function readStored<T extends object>(key: string, defaults: T): Partial<T> {
  let stored: unknown;
  try {
    stored = JSON.parse(localStorage.getItem(key) ?? "null");
  } catch {
    return {};
  }
  if (!stored || typeof stored !== "object") return {};

  const valid: Partial<T> = {};
  for (const field of Object.keys(defaults) as (keyof T)[]) {
    const value = (stored as Partial<T>)[field];
    if (typeof value === typeof defaults[field]) valid[field] = value;
  }
  return valid;
}
