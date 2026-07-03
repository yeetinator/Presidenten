import { crossfade } from "svelte/transition";
import { cubicOut } from "svelte/easing";
import { get } from "svelte/store";
import { fastForwardMode, state } from "../stores/gameStore";

const normalDuration = 500;

function getTransitionDuration() {
    return get(fastForwardMode) ? normalDuration / 2 : normalDuration;
}

export const [svelteSend, svelteReceive] = crossfade({
    duration: () => getTransitionDuration(),
    easing: cubicOut,
    fallback(node, params) {
        return {
            duration: getTransitionDuration(),
            css: (t) => `opacity: ${t}; transform: scale(${0.8 + 0.2 * t});`,
        };
    }
})

export const send = (node: HTMLElement, params: any) => {
    const result = svelteSend(node, params) as any;
    if (typeof result === "function") return () => {
        const obj = result();
        if (!obj) return obj;

        const sampleCss = obj.css ? obj.css(0.5, 0.5) : "";
        const isFlight = sampleCss.includes("translate");

        if (isFlight) return {
            ...obj,
            css: () => `opacity: 0; pointer-events: none;`,
        };
        else {
            if (params && params.isPile) {
                const currState = get(state);
                const pileIsReset = !currState || !currState.suit_last_move || currState.suit_last_move.length === 0;

                if (!pileIsReset) return {
                    ...obj,
                    css: () => `opacity: 1;`,
                }
            }
            return obj;
        }
    }
    return result;
}

export const receive = (node: HTMLElement, params: any) => {
    const result = svelteReceive(node, params) as any;
    if (typeof result === "function") return () => {
        const obj = result();
        if (!obj) return obj;

        const sampleCss = obj.css ? obj.css(0.5, 0.5) : "";
        const isFlight = sampleCss.includes("translate");

        if (isFlight) {
            const originalCss = obj.css;
            return {
                ...obj,
                css: (t: number, u: number) => {
                    const res = originalCss ? originalCss(t, u) : "";
                    return res.replace(/opacity:\s*[^;]+(;|$)/g, "") + "; opacity: 1;";
                }
            }
        }
        return obj;
    }
    return result;
}