<script lang="ts">
  export let suitCard: string = "3C";
  export let isFaceUp: boolean = true;
  export let isSelected: boolean = false;
  export let isBlinking: boolean = false;
  export let disabled: boolean = false;
  export let exchange: boolean = false;
  export let className: string = "";
  export let onClick: ((event: MouseEvent) => void) | undefined = undefined;

  $: imgSrc = isFaceUp
    ? `/cards/${suitCard.trim().toUpperCase()}.svg`
    : "/cards/1B.svg";
</script>

<button
  type="button"
  class={`relative h-28 w-20 select-none rounded-md bg-white shadow-md outline-none transition-all duration-200 ease-out md:h-36 md:w-24 ${className} ${
    disabled && !exchange
      ? ""
      : isSelected
        ? "-translate-y-6 scale-105 shadow-xl ring-4 ring-emerald-400"
        : "hover:-translate-y-2 hover:shadow-lg"
  } border ${isBlinking && !disabled ? "animate-jump-ready" : "border-slate-200"}`}
  {disabled}
  on:click={onClick}
>
  <img
    src={imgSrc}
    alt="Playing Card"
    class="h-full w-full object-fill rounded-md pointer-events-none"
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
      opacity: 0.35;
      transform: scale(0.98);
    }
    50% {
      opacity: 1;
      transform: scale(1.02);
    }
  }
  .animate-jump-ready::before {
    content: "";
    position: absolute;
    inset: -1px;
    border: 1px solid rgba(220, 38, 38, 1);
    box-shadow:
      0 0 0 5px rgba(220, 38, 38, 0.8),
      0 0 20px rgba(220, 38, 38, 0.6);
    border-radius: inherit;
    pointer-events: none;
    animation: red-flash 0.8s infinite ease-in-out;
    will-change: transform, opacity;
  }
</style>
