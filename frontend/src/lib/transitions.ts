import { crossfade } from "svelte/transition";
import type { TransitionConfig } from "svelte/transition";
import { cubicOut } from "svelte/easing";
import { get } from "svelte/store";
import { fastForwardMode, state } from "../stores/gameStore";

export const NORMAL_DURATION_MS = 200;
export const FAST_FORWARD_DURATION_MS = NORMAL_DURATION_MS / 2;

export interface CardTransitionParams {
    key: string;
    isPile?: boolean;
}

type DeferredTransition = (node: Element, params: CardTransitionParams) => TransitionConfig | (() => TransitionConfig);

export function getTransitionDuration() {
    return get(fastForwardMode) ? FAST_FORWARD_DURATION_MS : NORMAL_DURATION_MS;
}

let usedFallback = false;

const [svelteSend, svelteReceive] = crossfade({
    duration: getTransitionDuration,
    easing: cubicOut,
    fallback(): TransitionConfig {
        usedFallback = true;
        return {
            duration: getTransitionDuration(),
            css: (t) => `opacity: ${t}; transform: scale(${0.8 + 0.2 * t});`,
        };
    }
})

function withFlightDetection(
    transition: DeferredTransition,
    adapt: (config: TransitionConfig, isFlight: boolean, params: CardTransitionParams) => TransitionConfig
) {
    return (node: Element, params: CardTransitionParams) => {
        const result = transition(node, params);
        if (typeof result !== "function") return adapt(result, false, params);

        return () => {
            usedFallback = false;
            const config = result();
            return adapt(config, !usedFallback, params);
        }
    }
}

function pileIsReset() {
    const currState = get(state);
    return !currState?.suit_last_move?.length;
}

export const send = withFlightDetection(svelteSend, (config, isFlight, params) => {
    if (isFlight) return { ...config, css: () => `opacity: 0; pointer-events: none;` };
    if (params.isPile && !pileIsReset()) return { ...config, css: () => `opacity: 1;` };
    return config;
});


export const receive = withFlightDetection(svelteReceive, (config, isFlight) => {
    if (!isFlight) return config;

    const { css } = config;
    return {
        ...config,
        css: (t, u) => `${css ? css(t, u) : ""}; opacity: 1;`
    };
});