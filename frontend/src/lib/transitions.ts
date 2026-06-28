import { crossfade } from "svelte/transition";
import { cubicOut } from "svelte/easing";
import { get } from "svelte/store";
import { state } from "../stores/gameStore";

export const [svelteSend, svelteReceive] = crossfade({
    duration: 500,
    easing: cubicOut,
    fallback(node, params) {
        return {
            duration: 500,
            css: (t) => `opacity: ${t}; transform: scale(${0.8 + 0.2 * t});`,
        };
    }
})

export const send = (node: HTMLElement, params: any) => {
    const result = svelteSend(node, params) as any;
    if (typeof result === "function") return () => {
        const obj = result();
        return {
            ...obj,
            css: () => `opacity: 0; pointer-events: none;`,
        };
    }
    return result;
}

export const receive = (node: HTMLElement, params: any) => {
    const result = svelteReceive(node, params) as any;
    if (typeof result === "function") return () => {
        const obj = result();
        const originalCss = obj.css;
        return {
            ...obj,
            css: (t: number, u: number) => {
                const res = originalCss ? originalCss(t, u) : "";
                return res.replace(/opacity:\s*[^;]+(;|$)/g, "") + "; opacity: 1;";
            },
        };
    }
    return result;
}