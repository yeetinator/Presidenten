import { crossfade } from "svelte/transition";
import { cubicOut } from "svelte/easing";

export const [send, receive] = crossfade({
    duration: 500,
    easing: cubicOut,
    fallback(node, params) {
        return {
            duration: 500,
            css: (t) => `opacity: ${t}; transform: scale(${0.8 + 0.2 * t});`,
        };
    }
})