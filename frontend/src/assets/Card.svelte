<script lang="ts">
  export let suitCard: string = "3C";
  export let isFaceUp: boolean = true;
  export let isSelected: boolean = false;
  export let isBlinking: boolean = false;
  export let disabled: boolean = false;
  export let className: string = "";
  export let onClick: ((event: MouseEvent) => void) | undefined = undefined;

  $: imgSrc = isFaceUp
    ? `/cards/${suitCard.trim().toUpperCase()}.svg`
    : "/cards/1B.svg";
</script>

<button
  type="button"
  class={`relative h-28 w-20 select-none rounded-xl bg-white shadow-md outline-none transition-all duration-200 ease-out md:h-36 md:w-24 ${className} ${
    isSelected
      ? "-translate-y-6 scale-105 shadow-xl ring-4 ring-emerald-400"
      : "hover:-translate-y-2 hover:shadow-lg"
  } border border-slate-200 ${isBlinking ? "animate-jump-ready" : ""}`}
  {disabled}
  on:click={onClick}
>
  <div class="w-full h-full overflow-hidden rounded-xl pointer-events-none">
    <img src={imgSrc} alt="Playing Card" class="h-full w-full object-contain" />
  </div>
</button>

<style>
  button::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: -6px;
    right: -6px;
    border-radius: inherit;
  }
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
