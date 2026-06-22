<script lang="ts">
  export let value: number;
  export let suit: "hearts" | "diamonds" | "clubs" | "spades" = "clubs";
  export let isFaceUp: boolean = true;
  export let isSelected: boolean = false;
  export let isBlinking: boolean = false;

  function getCardImage(val: number, cardSuit: string): string {
    if (!isFaceUp) return "/cards/1B.svg";

    let rankStr = String(val);
    if (val === 11) rankStr = "J";
    else if (val === 12) rankStr = "Q";
    else if (val === 13) rankStr = "K";
    else if (val === 14) rankStr = "A";
    else if (val === 15) rankStr = "2";

    const suitLetter = cardSuit.charAt(0).toUpperCase();
    return `/cards/${rankStr}${suitLetter}.svg`;
  }

  $: imgSrc = getCardImage(value, suit);
</script>

<button
  type="button"
  class="relative w-20 h-28 md:w-24 md:h-36 rounded-xl overflow-hidden bg-white transition-all duration-200 ease-out shadow-md select-none outline-none {isSelected
    ? '-translate-y-6 shadow-xl ring-4 ring-emerald-400 scale-105'
    : 'hover:-translate-y-2 hover:shadow-lg'} {isBlinking
    ? 'animate-jump-ready'
    : 'border border-slate-200'}"
  on:click
  ><img
    src={imgSrc}
    alt="Playing Card"
    class="w-full h-full object-contain pointer-events-none"
  /></button
>

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
