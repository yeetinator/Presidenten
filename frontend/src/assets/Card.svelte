<script lang="ts">
  export let suitCard: string = "3C";
  export let isFaceUp: boolean = true;
  export let isSelected: boolean = false;
  export let isBlinking: boolean = false;
  export let disabled: boolean = false;
  export let className: string = "";
  export let onClick: ((event: MouseEvent) => void) | undefined = undefined;

  function getCardImage(cardCode: string): string {
    if (!isFaceUp) return "/cards/1B.svg";

    const normalizedCode = cardCode.trim().toUpperCase();
    return `/cards/${normalizedCode}.svg`;
  }

  $: imgSrc = getCardImage(suitCard);
</script>

<button
  type="button"
  class={`relative h-28 w-20 select-none overflow-hidden rounded-xl bg-white shadow-md outline-none transition-all duration-200 ease-out md:h-36 md:w-24 ${className} ${
    isSelected
      ? "-translate-y-6 scale-105 shadow-xl ring-4 ring-emerald-400"
      : "hover:-translate-y-2 hover:shadow-lg"
  } border border-slate-200 ${isBlinking ? "animate-jump-ready" : ""}`}
  {disabled}
  on:click={onClick}
>
  <img
    src={imgSrc}
    alt="Playing Card"
    class="pointer-events-none h-full w-full object-contain"
  />
</button>

<style>
  @keyframes red-flash {
    0%,
    100% {
      border-color: rgba(239, 68, 68, 0.4);
      box-shadow: 0 0 0px rgba(239, 68, 68, 0);
    }
    50% {
      border-color: rgba(220, 38, 38, 1);
      box-shadow: 0 0 15px rgba(220, 38, 38, 0.8);
    }
  }
  :global(.animate-jump-ready) {
    animation: red-flash 0.6s infinite ease-in-out !important;
    border-width: 3px !important;
  }
</style>
