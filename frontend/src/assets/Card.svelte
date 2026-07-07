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
  class={`relative h-28 w-20 select-none rounded-md bg-white shadow-md outline-none transition-all duration-200 ease-out md:h-36 md:w-24 ${className} ${
    disabled
      ? ""
      : isSelected
        ? "-translate-y-6 scale-105 shadow-xl ring-4 ring-emerald-400"
        : "hover:-translate-y-2 hover:shadow-lg"
  } border border-slate-200 overflow-hidden ${isBlinking && !disabled ? "animate-jump-ready" : ""}`}
  {disabled}
  on:click={onClick}
>
  <img
    src={imgSrc}
    alt="Playing Card"
    class="h-full w-full object-fill pointer-events-none"
  />
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
      opacity: 0.2;
    }
    50% {
      opacity: 1;
    }
  }
  :global(.animate-jump-ready)::before {
    content: "";
    position: absolute;
    inset: -3px;
    border: 3px solid rgba(220, 38, 38, 1);
    box-shadow: 0 0 15px rgba(220, 38, 38, 0.8);
    border-radius: inherit;
    pointer-events: none;
    z-index: 10;
    animation: red-flash 0.6s infinite ease-in-out;
  }
</style>
