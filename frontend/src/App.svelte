<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import Card from "./assets/Card.svelte";
  import Lobby from "./lib/components/Lobby.svelte";
  import Rules from "./assets/Rules.svelte";
  import OpponentSeat from "./lib/components/OpponentSeat.svelte";
  import SummaryModal from "./lib/components/SummaryModal.svelte";
  import {
    gameStore,
    exchangePrompt,
    jumpInPrompt,
    fastForwardMode,
    roundSummary,
    revealedBotSeat,
    selectedCards,
    isDealing,
    inGame,
    state as gameState,
    currTheme,
    type GameStateUpdate,
  } from "./stores/gameStore";
  import {
    send,
    receive,
    NORMAL_DURATION_MS,
    FAST_FORWARD_DURATION_MS,
  } from "./lib/transitions";
  import { fade } from "svelte/transition";
  import type { OpponentView } from "./lib/types";

  const websocketUrl =
    import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/game";

  type MoveLabelTone = "waiting" | "prompt" | "ready" | "invalid" | "exchange";
  type MoveLabel = { text: string; tone: MoveLabelTone };

  const MOVE_LABEL_TONES: Record<MoveLabelTone, string> = {
    waiting: "text-[0.65rem] font-semibold tracking-[0.22em] text-white/35",
    prompt: "text-[0.7rem] font-bold tracking-[0.16em] text-white/85",
    ready: "text-[0.75rem] font-black tracking-[0.16em] animate-label-in",
    invalid: "text-[0.75rem] font-black tracking-[0.16em] text-red-400",
    exchange: "text-[0.75rem] font-black tracking-[0.16em] text-amber-300",
  };

  let deck: string[] = [];
  let visibleCardsPerSeat: Record<number, string[]> = {};
  let isAnimatingDeal = false;

  $: if ($isDealing && !isAnimatingDeal && $gameState) {
    runDealSequence();
  }

  $: transitionMs = $fastForwardMode
    ? FAST_FORWARD_DURATION_MS
    : NORMAL_DURATION_MS;
  $: suitedHand = $isDealing
    ? (visibleCardsPerSeat[0] ?? [])
    : ($gameState?.suited_hand ?? []);
  $: exchangeRequiredCards = $exchangePrompt?.requiredCards ?? 0;
  $: exchangeCanChoose = $exchangePrompt?.canChoose ?? false;
  $: isMyTurn = $gameState?.curr_turn === 0;
  $: haveIPassed = ($gameState?.passed?.includes(0) && !isMyTurn) ?? false;
  $: jumpInVisible = !!$jumpInPrompt && !!$gameState?.is_finish_prompt;
  $: jumpInAutoMove =
    jumpInVisible && $gameState ? gameStore.getAutoFinishMove() : null;
  $: selectedMoveValues = $selectedCards.map(gameStore.stripSuitCard);
  $: isSelectionLegal = (() => {
    if (
      !$gameState?.legal_moves_suits ||
      $exchangePrompt ||
      !$selectedCards.length
    )
      return false;
    const sortedSelection = [...selectedMoveValues].sort();
    return $gameState.legal_moves_suits.some((legalMove) =>
      gameStore.arraysEqual([...legalMove].sort(), sortedSelection),
    );
  })();

  let enableCards = false;
  $: if ($gameState || $exchangePrompt || $jumpInPrompt) {
    const needPlayerInput = isMyTurn || !!$exchangePrompt || jumpInVisible;
    if (needPlayerInput) enableCards = true;
  }

  $: isExchangeReady =
    !!$exchangePrompt && $selectedCards.length === exchangeRequiredCards;
  $: isCenterClickable = (isMyTurn && isSelectionLegal) || isExchangeReady;

  $: opponentViews = Object.entries($gameState?.player_roles ?? {})
    .map(([seatStr, role]) => {
      const seat = Number(seatStr);
      return { seat, role };
    })
    .filter((opponent) => opponent.seat !== 0)
    .sort((a, b) => a.seat - b.seat)
    .map((opponent, index, arr) => ({
      ...opponent,
      suitedHand: $isDealing
        ? (visibleCardsPerSeat[opponent.seat] ?? [])
        : ($gameState?.opp_suited_hands?.[opponent.seat] ?? []),
      label: `${opponent.role || "Waiting"} ${$gameState?.player_types?.[opponent.seat] ?? "Bot"}`,
      position:
        index === 0 ? "left" : index === arr.length - 1 ? "right" : "top",
      is_turn:
        $gameState?.resume_turn === opponent.seat ||
        ($gameState?.curr_turn === opponent.seat && !$gameState?.resume_turn),
      has_passed: $gameState?.passed?.includes(opponent.seat) ?? false,
    })) as OpponentView[];

  $: topOpponents = opponentViews.filter((o) => o.position === "top");
  $: leftOpponent = opponentViews.find((o) => o.position === "left");
  $: rightOpponent = opponentViews.find((o) => o.position === "right");
  $: selectedMoveLabel = getMoveLabel({
    state: $gameState,
    selected: $selectedCards,
    isExchange: !!$exchangePrompt,
    myTurn: isMyTurn,
    legal: isSelectionLegal,
    reqCards: exchangeRequiredCards,
    canChoose: exchangeCanChoose,
  });

  type PileCard = { suitCard: string; rotation: number };
  let visualPile: PileCard[][] = [];
  $: {
    const currSuits = $gameState?.suit_last_move ?? [];
    if (currSuits.length === 0) visualPile = [];
    else {
      const lastPlay = visualPile[visualPile.length - 1] ?? [];
      const isAlreadyAdded =
        lastPlay.length === currSuits.length &&
        currSuits.every((suitCard, idx) => suitCard === lastPlay[idx].suitCard);
      const isExtension =
        !isAlreadyAdded &&
        lastPlay.length > 0 &&
        currSuits.length > lastPlay.length &&
        lastPlay.every((card, idx) => card.suitCard === currSuits[idx]);

      if (!isAlreadyAdded) {
        const play = buildPilePlay(
          currSuits,
          isExtension ? lastPlay : [],
          $gameState?.pile_leader ?? 0,
          opponentViews,
        );
        visualPile = isExtension
          ? [...visualPile.slice(0, -1), play]
          : [...visualPile, play];
      }
    }
  }
  $: flatVisualPile = visualPile.flatMap((play, playIndex) =>
    play.map((card, cardIndex) => ({
      ...card,
      playIndex,
      cardIndex,
      playLength: play.length,
    })),
  );
  $: isTurnPending =
    isMyTurn &&
    !$gameState?.can_pass &&
    ($gameState?.legal_moves_suits?.length ?? 0) === 0;

  let lastStateRef: typeof $gameState = null;
  $: if ($gameState && !$exchangePrompt) {
    if ($gameState !== lastStateRef) {
      lastStateRef = $gameState;
      gameStore.clearSelectedCards();
    }
  }

  onMount(() => {
    gameStore.connect(websocketUrl);
  });

  onDestroy(() => {
    gameStore.disconnect();
  });

  function getMoveLabel({
    state,
    selected,
    isExchange,
    myTurn,
    legal,
    reqCards,
    canChoose,
  }: {
    state: GameStateUpdate | null;
    selected: string[];
    isExchange: boolean;
    myTurn: boolean;
    legal: boolean;
    reqCards: number;
    canChoose: boolean;
  }): MoveLabel {
    if (isExchange) {
      if (reqCards === 0)
        return {
          text: "Citizens are exempt from exchanging",
          tone: "waiting",
        };

      if (selected.length === reqCards)
        return {
          text: canChoose
            ? `${reqCards} cards are ready to send`
            : `Highest ${reqCards} card${selected.length > 1 ? "s" : ""} pre-selected`,
          tone: "exchange",
        };

      if (selected.length > 0)
        return {
          text: `Selected ${selected.length}/${reqCards} cards`,
          tone: "prompt",
        };

      return { text: `Select ${reqCards} cards to give away`, tone: "prompt" };
    }

    if (!state) return { text: "Waiting for the deal", tone: "waiting" };

    if (state.suited_hand?.length === 0)
      return { text: "You are out for this round", tone: "waiting" };

    if (!myTurn) return { text: "Waiting for your turn", tone: "waiting" };

    if (state.legal_moves_suits?.length === 0 && !state.can_pass)
      return {
        text: state.suit_last_move?.length
          ? "Pile clears · you lead next"
          : "You lead next",
        tone: "waiting",
      };

    if (state.legal_moves_suits.length === 0 && state.can_pass)
      return { text: "No legal moves · you must pass", tone: "invalid" };

    if (selected.length > 0)
      return legal
        ? {
            text: `Valid Move · ${selected.length} card${selected.length > 1 ? "s" : ""}`,
            tone: "ready",
          }
        : { text: "Illegal Move", tone: "invalid" };

    return {
      text: "Click cards from your hand to queue them for play",
      tone: "prompt",
    };
  }

  function getPileBaseAngle(leaderSeat: number, opponents: OpponentView[]) {
    if (leaderSeat === 0) return 0;

    const position = opponents.find((o) => o.seat === leaderSeat)?.position;
    if (position === "left") return 90;
    if (position === "top") return 180;
    if (position === "right") return 270;
    return 0;
  }

  function buildPilePlay(
    suitCards: string[],
    previousCards: PileCard[],
    leaderSeat: number,
    opponents: OpponentView[],
  ): PileCard[] {
    const baseAngle = getPileBaseAngle(leaderSeat, opponents);
    const groupPlayJitter = Math.random() * 70 - 35;

    return suitCards.map((suitCard) => {
      const existing = previousCards.find((c) => c.suitCard === suitCard);
      if (existing) return existing;

      const microjitter = Math.random() * 10 - 5;
      return {
        suitCard,
        rotation: baseAngle + groupPlayJitter + microjitter,
      };
    });
  }

  function getHorizontalOverlapStyle(index: number) {
    return index === 0 ? "" : "margin-left: -4rem;";
  }

  async function handleStartGame(config: {
    numPlayers: number;
    numRounds: number;
    playerTypes: number[];
  }) {
    try {
      gameStore.clearRoundSummary();
      gameStore.clearExchangePrompt();
      gameStore.clearSelectedCards();
      gameStore.clearFastForwardMode();
      await gameStore.startGame({
        type: "START_GAME",
        num_players: config.numPlayers,
        num_rounds: config.numRounds,
        player_types: config.playerTypes,
      });
    } catch (error) {
      console.error(error);
    }
  }

  function handleCenterAction() {
    if (isExchangeReady) handleConfirmExchange();
    else if (isMyTurn && isSelectionLegal) handlePlay();
  }

  async function runDealSequence() {
    if (!$gameState) return;
    isAnimatingDeal = true;

    const rawHands: Record<number, string[]> = {
      0: $gameState.suited_hand ?? [],
      ...($gameState.opp_suited_hands ?? {}),
    };
    const allCards: string[] = [];
    visibleCardsPerSeat = {};

    for (const [seatStr, hand] of Object.entries(rawHands)) {
      const seat = Number(seatStr);
      visibleCardsPerSeat[seat] = [];
      allCards.push(...hand);
    }
    deck = allCards;

    const seats = Object.keys(rawHands)
      .map(Number)
      .sort((a, b) => a - b);
    const maxHandSize = Math.max(
      ...Object.values(rawHands).map((h) => h.length),
      0,
    );
    const dealQueue: { seat: number; card: string }[] = [];

    for (let cardIdx = 0; cardIdx < maxHandSize; cardIdx++)
      for (const seat of seats)
        if (rawHands[seat][cardIdx])
          dealQueue.push({ seat, card: rawHands[seat][cardIdx] });

    const delayMs = 55;
    for (const step of dealQueue) {
      deck = deck.filter((c) => c !== step.card);
      visibleCardsPerSeat[step.seat] = [
        ...(visibleCardsPerSeat[step.seat] ?? []),
        step.card,
      ];

      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
    try {
      await gameStore.dealtCards();
    } catch (error) {
      console.error(error);
    } finally {
      isAnimatingDeal = false; // or the next game's deal would never start
    }
  }

  async function handlePlay() {
    try {
      enableCards = false;
      await gameStore.playSelectedCards();
    } catch (error) {
      console.error(error);
    }
  }

  async function handlePass() {
    try {
      enableCards = false;
      await gameStore.passTurn();
    } catch (error) {
      console.error(error);
    }
  }

  async function handleJumpIn(finishMove: string[]) {
    try {
      enableCards = false;
      await gameStore.playJumpInPrompt(finishMove);
    } catch (error) {
      console.error(error);
    }
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (!$inGame || event.repeat || $roundSummary) return;

    const target = event.target as HTMLElement | null;
    if (
      target?.isContentEditable ||
      ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName ?? "")
    )
      return;

    if (event.code === "Space") {
      if (!jumpInVisible || !jumpInAutoMove?.length) return;
      event.preventDefault();
      handleJumpIn(jumpInAutoMove[0]);
      return;
    }

    if (event.code === "Enter" && isCenterClickable) {
      event.preventDefault();
      handleCenterAction();
    }
  }

  async function handleNextRound() {
    try {
      await gameStore.nextRound();
    } catch (error) {
      console.error(error);
    }
  }

  async function handleConfirmExchange() {
    if (!$gameState || !$exchangePrompt) return;

    const suitCards = $selectedCards;
    if (suitCards.length !== exchangeRequiredCards) return;

    try {
      enableCards = false;
      await gameStore.sendExchangeCards(suitCards);
      gameStore.clearSelectedCards();
      gameStore.clearExchangePrompt();
    } catch (error) {
      console.error(error);
    }
  }

  async function handleFastForward() {
    try {
      await gameStore.fastForwardGame();
    } catch (error) {
      console.error(error);
    }
  }

  function handleQuit() {
    gameStore.clearAll();
    gameStore.disconnect();
    gameStore.connect(websocketUrl);
  }

  function handleToggleCard(suitCard: string) {
    if ($exchangePrompt && !exchangeCanChoose && exchangeRequiredCards > 0)
      return;
    if (!$exchangePrompt && !isMyTurn) return;

    const clickedCardValue = gameStore.stripSuitCard(suitCard);
    const finishMove = gameStore.getAutoFinishMove(clickedCardValue);

    if (jumpInVisible && finishMove && clickedCardValue !== null) {
      handleJumpIn(finishMove);
      return;
    }

    if ($selectedCards.includes(suitCard)) {
      gameStore.selectedCards.set(
        $selectedCards.filter((selected) => selected !== suitCard),
      );
      return;
    }

    const selectionLimit = $exchangePrompt ? exchangeRequiredCards : 4;
    if ($selectedCards.length >= selectionLimit) return;
    gameStore.selectedCards.set([...$selectedCards, suitCard]);
  }
</script>

<svelte:window on:keydown={handleKeyDown} />

{#if !$inGame}
  <Lobby onStartGame={handleStartGame} />
{:else}
  <main
    class="h-screen max-h-screen overflow-hidden p-2 text-white md:p-3"
    style={`
      background: ${$currTheme.mainBg};
      --bg-felt: ${$currTheme.feltBg};
      --border-felt: ${$currTheme.feltBorder};
      --accent-border: ${$currTheme.accentBorder};
      --accent-bg: ${$currTheme.accentBg};
      --accent-text: ${$currTheme.accentText};
      --accent-shadow: ${$currTheme.accentShadow};
      --card-ring: ${$currTheme.cardRing};
      --jump-color: ${$currTheme.jumpInColor};
    `}
  >
    {#if jumpInVisible}
      <div
        aria-hidden="true"
        transition:fade={{ duration: 150 }}
        class="animate-jump-vignette pointer-events-none fixed inset-0 z-40"
      ></div>
    {/if}

    {#if !$roundSummary}
      <Rules className="left-4" side="left" />
      <button
        class="fixed right-5 top-4 z-30 rounded-md border border-red-300/30 bg-red-500 px-3 py-1 text-[0.65rem]
        font-black tracking-[0.28em] text-white shadow-lg shadow-black/30 transition hover:bg-red-400 active:scale-[0.98]"
        type="button"
        on:click={handleQuit}
      >
        QUIT
      </button>

      {#if suitedHand.length === 0 && opponentViews.some((o) => o.suitedHand.length !== 0) && !$exchangePrompt}
        <button
          class="fixed right-22 top-4 z-30 rounded-md border border-blue-300/30 bg-blue-500 px-3 py-1 text-[0.65rem]
        font-black tracking-[0.28em] text-white shadow-lg shadow-black/30 transition hover:bg-blue-400 active:scale-[0.98]"
          type="button"
          on:click={handleFastForward}
        >
          {#if $fastForwardMode}
            <div in:fade={{ duration: 150 }} out:fade={{ duration: 150 }}>
              <span class="animate-arrow-pulse opacity-20">&gt;</span>
              <span
                class="animate-arrow-pulse opacity-20 [animation-delay:0.15s]"
                >&gt;</span
              >
              <span
                class="animate-arrow-pulse opacity-20 [animation-delay:0.3s]"
                >&gt;</span
              >
            </div>
          {:else}
            <span in:fade={{ duration: 150 }} out:fade={{ duration: 150 }}
              >FAST FORWARD</span
            >
          {/if}
        </button>
      {/if}
    {/if}

    <section
      class="mx-auto grid h-full max-h-full grid-rows-[1fr_auto] gap-2 max-w-screen-2xl"
    >
      <section
        class="rounded-2xl border p-1 shadow-lg backdrop-blur-md flex flex-col justify-between overflow-hidden"
        style="background: var(--bg-felt); border-color: var(--border-felt);"
      >
        <div
          class="grid gap-3 xl:grid-cols-[16rem_1fr_16rem] xl:grid-rows-[auto_1fr] h-full items-center"
        >
          <div class="xl:col-span-3">
            {#if topOpponents.length > 0}
              <div class="flex flex-wrap justify-center gap-3 overflow-visible">
                {#each topOpponents as opponent (opponent.seat)}
                  <OpponentSeat
                    {opponent}
                    revealBotCards={$revealedBotSeat === opponent.seat}
                    className="w-60 shrink-0 px-3 py-1.5"
                  />
                {/each}
              </div>
            {/if}
          </div>

          <aside class="flex items-center justify-center xl:row-start-2">
            {#if leftOpponent}
              <OpponentSeat
                opponent={leftOpponent}
                revealBotCards={$revealedBotSeat === leftOpponent.seat}
                className="w-full max-w-56 p-3"
              />
            {/if}
          </aside>

          <div class="flex items-center justify-center xl:row-start-2">
            {#if !$isDealing}
              <button
                type="button"
                class="w-full max-w-xs aspect-square flex flex-col justify-between p-4 relative rounded-3xl border transition-all duration-300 outline-none
                  {isExchangeReady
                  ? 'border-amber-400 bg-amber-500/5 shadow-[0_0_30px_rgba(245,158,11,0.25)] hover:bg-amber-500/10 cursor-pointer active:scale-[0.99]'
                  : isMyTurn && isSelectionLegal
                    ? 'cursor-pointer active:scale-[0.99]'
                    : isMyTurn && !$exchangePrompt
                      ? 'animate-turn-glow cursor-default'
                      : 'border-white/5 bg-white/1 cursor-default'}"
                style={(isMyTurn && isSelectionLegal) ||
                (isMyTurn && !$exchangePrompt)
                  ? `border-color: var(--accent-border); background-color: var(--accent-bg); ${isSelectionLegal ? "box-shadow: var(--accent-shadow);" : ""}`
                  : ""}
                disabled={!isCenterClickable}
                on:click={handleCenterAction}
              >
                <div
                  class="w-full text-center pointer-events-none z-10 select-none"
                >
                  {#if isExchangeReady}
                    <span
                      class="text-[0.6rem] font-black uppercase tracking-[0.25em] text-amber-400 animate-pulse"
                      >➔ Click Here to Send Cards
                    </span>
                  {:else if isMyTurn && isSelectionLegal}
                    <span
                      class="text-[0.6rem] font-black uppercase tracking-[0.25em] animate-pulse"
                      style="color: var(--accent-text);"
                      >➔ Click Here to Play Move
                    </span>
                  {:else if $selectedCards.length > 0 && !isSelectionLegal && !$exchangePrompt}
                    <span
                      class="text-[0.6rem] font-bold uppercase tracking-[0.25em] text-red-400/90"
                      >✕ Invalid Combination</span
                    >
                  {:else}
                    <span
                      class="text-[0.55rem] font-bold uppercase tracking-[0.25em] text-white/20"
                      >Game Pile</span
                    >
                  {/if}
                </div>
                <div
                  class="absolute inset-0 flex items-center justify-center overflow-visible pointer-events-none"
                >
                  <div
                    class="relative flex items-center justify-center overflow-visible h-28 w-full"
                  >
                    {#each flatVisualPile as card (card.suitCard)}
                      <div
                        in:receive={{ key: card.suitCard }}
                        out:send={{ key: card.suitCard, isPile: true }}
                        on:introstart={gameStore.startAnimation}
                        on:introend={gameStore.endAnimation}
                        on:outrostart={gameStore.startAnimation}
                        on:outroend={gameStore.endAnimation}
                        class="absolute transition-all duration-300 flex justify-center origin-center"
                        style={`transition-duration: ${transitionMs}ms; z-index: ${card.playIndex * 10 + card.cardIndex}; 
                      transform: translateX(${(card.cardIndex - (card.playLength - 1) / 2) * 1.4}rem);`}
                      >
                        <div
                          style={`transform: rotate(${card.rotation}deg);`}
                          class="origin-center flex justify-center items-center"
                        >
                          <Card
                            suitCard={card.suitCard}
                            isFaceUp={true}
                            disabled={true}
                            className="shrink-0 scale-[0.85] origin-center shadow-lg shadow-black/40"
                          />
                        </div>
                      </div>
                    {/each}
                  </div>
                </div>
                <div
                  class="w-full min-h-10 flex items-end justify-center text-center pointer-events-none z-10 select-none"
                  aria-live="polite"
                >
                  <span
                    class="max-w-56 mx-auto uppercase leading-tight transition-colors duration-200 drop-shadow-[0_2px_6px_rgba(0,0,0,0.65)] {MOVE_LABEL_TONES[
                      selectedMoveLabel.tone
                    ]}"
                    style={selectedMoveLabel.tone === "ready"
                      ? "color: var(--accent-text); text-shadow: 0 0 14px var(--accent-text);"
                      : ""}>{selectedMoveLabel.text}</span
                  >
                </div>
              </button>
            {:else}
              {#each deck as suitCard (suitCard)}
                <div
                  in:receive={{ key: suitCard }}
                  out:send={{ key: suitCard }}
                  on:introstart={gameStore.startAnimation}
                  on:introend={gameStore.endAnimation}
                  on:outrostart={gameStore.startAnimation}
                  on:outroend={gameStore.endAnimation}
                  class="absolute transition-all duration-300 flex justify-center origin-center"
                  style={`transition-duration: ${transitionMs}ms;`}
                >
                  <Card
                    {suitCard}
                    isFaceUp={false}
                    disabled={true}
                    className="shrink-0 scale-[0.85] origin-center shadow-lg shadow-black/40"
                  />
                </div>
              {/each}
            {/if}
          </div>

          <aside class="flex items-center justify-center xl:row-start-2">
            {#if rightOpponent}
              <OpponentSeat
                opponent={rightOpponent}
                revealBotCards={$revealedBotSeat === rightOpponent.seat}
                className="w-full max-w-56 p-3"
              />
            {/if}
          </aside>
        </div>

        <div class="w-full flex flex-col gap-1 mt-auto">
          <div
            class="h-9 flex items-center justify-center gap-2 z-20 overflow-visible"
          >
            {#if jumpInVisible && jumpInAutoMove?.length}
              <button
                transition:fade={{ duration: 100 }}
                type="button"
                class="animate-jump-cta rounded-full border-2 bg-black/60 backdrop-blur-md px-6 py-1.5
          text-[0.7rem] font-black uppercase tracking-[0.22em] transition-transform active:scale-95"
                style="border-color: var(--jump-color); color: var(--jump-color);"
                on:click={() =>
                  jumpInAutoMove && handleJumpIn(jumpInAutoMove[0])}
              >
                ⚡ {$jumpInPrompt?.message ?? "Jump In"}
                <span class="opacity-70">· Space</span></button
              >
            {:else if isMyTurn && !$exchangePrompt && !isTurnPending && !$isDealing}
              <span
                transition:fade={{ duration: 100 }}
                class="animate-turn-pill rounded-full border px-4 py-1.5 text-[0.65rem] font-black uppercase tracking-[0.22em] backdrop-blur-md"
                style="border-color: var(--accent-border); background: var(--accent-bg); color: var(--accent-text);"
                >Your Turn</span
              >
            {/if}
            {#if isMyTurn && $gameState?.can_pass && $gameState?.pile_leader !== 0 && !isTurnPending}
              <button
                transition:fade={{ duration: 100 }}
                type="button"
                class="rounded-full border border-white/20 bg-black/50 backdrop-blur-md px-6 py-1.5 text-[0.65rem] font-black uppercase tracking-[0.2em] text-white/90
                shadow-md transition-all hover:bg-white/10 hover:text-white active:scale-95"
                on:click={handlePass}>Pass Turn</button
              >
            {/if}
          </div>
          <div
            class="px-3 pt-6 overflow-visible relative flex items-center justify-center"
          >
            <div
              aria-hidden="true"
              class="pointer-events-none absolute inset-0 transition-opacity duration-500 {isMyTurn &&
              !haveIPassed
                ? 'opacity-100'
                : 'opacity-0'}"
              style="background: radial-gradient(ellipse 55% 130% at 50% 135%, var(--accent-bg), transparent 70%);"
            ></div>
            <div
              class="flex w-full justify-center overflow-visible h-28 md:h-36 items-center transition-all duration-300 {haveIPassed
                ? 'opacity-40 grayscale pointer-events-none'
                : ''}"
              style={`transition-duration: ${transitionMs}ms;`}
            >
              {#each suitedHand as suitCard, index (suitCard)}
                <div
                  in:receive={{ key: suitCard }}
                  out:send={{ key: suitCard }}
                  on:introstart={gameStore.startAnimation}
                  on:introend={gameStore.endAnimation}
                  on:outrostart={gameStore.startAnimation}
                  on:outroend={gameStore.endAnimation}
                  class="relative pointer-events-none transition-all duration-300"
                  style={`transition-duration: ${transitionMs}ms; ${getHorizontalOverlapStyle(index)}`}
                >
                  <Card
                    {suitCard}
                    isFaceUp={true}
                    isSelected={$selectedCards.includes(suitCard)}
                    isBlinking={!!(
                      jumpInVisible &&
                      jumpInAutoMove?.some((arr) =>
                        arr.includes(gameStore.stripSuitCard(suitCard)),
                      )
                    )}
                    disabled={haveIPassed ||
                      (!$exchangePrompt && !isMyTurn) ||
                      (!!$exchangePrompt &&
                        (!exchangeCanChoose || exchangeRequiredCards === 0)) ||
                      !enableCards ||
                      isTurnPending}
                    className="shrink-0 scale-[0.75] md:scale-[0.95] pointer-events-auto"
                    onClick={() => handleToggleCard(suitCard)}
                  />
                </div>
              {/each}
            </div>
          </div>
        </div>
      </section>

      {#if $roundSummary}
        <SummaryModal
          eyebrow="Round complete"
          eyebrowClass="text-amber-200/75"
          title="Play the next round or quit"
          scores={$roundSummary.scores}
          playerTypes={$gameState?.player_types ?? {}}
        >
          <div slot="actions" class="flex gap-3">
            <button
              class="rounded-xl bg-emerald-400 px-5 py-3 font-semibold text-emerald-950 transition hover:bg-emerald-300"
              type="button"
              on:click={handleNextRound}
            >
              Play Next Round
            </button>
            <button
              class="rounded-xl border border-white/10 bg-white/10 px-5 py-3 font-semibold text-white transition hover:bg-white/15"
              type="button"
              on:click={handleQuit}
            >
              Quit
            </button>
          </div>
        </SummaryModal>
      {/if}
    </section>
  </main>
{/if}

<style>
  @keyframes arrow-pulse {
    0%,
    100% {
      opacity: 0.2;
      transform: translateX(0);
    }
    50% {
      opacity: 1;
      transform: translateX(0.2rem);
    }
  }
  :global(.animate-arrow-pulse) {
    animation: arrow-pulse 0.6s infinite ease-in-out;
  }

  @keyframes turn-glow {
    0%,
    100% {
      box-shadow: 0 0 0 0 rgba(255, 255, 255, 0);
    }
    50% {
      box-shadow: var(--accent-shadow);
    }
  }
  :global(.animate-turn-glow) {
    animation: turn-glow 2s infinite ease-in-out;
  }

  @keyframes turn-pill {
    0%,
    100% {
      opacity: 0.75;
      transform: scale(0.99);
    }
    50% {
      opacity: 1;
      transform: scale(1.02);
    }
  }
  :global(.animate-turn-pill) {
    animation: turn-pill 1.6s infinite ease-in-out;
  }

  @keyframes jump-cta {
    0%,
    100% {
      box-shadow: 0 0 0 0 var(--jump-color);
      transform: scale(1);
    }
    50% {
      box-shadow:
        0 0 18px 2px var(--jump-color),
        0 0 40px -8px var(--jump-color);
      transform: scale(1.04);
    }
  }
  :global(.animate-jump-cta) {
    animation: jump-cta 0.8s infinite ease-in-out;
  }

  @keyframes jump-vignette {
    0%,
    100% {
      box-shadow: inset 0 0 60px -25px var(--jump-color);
      opacity: 0.5;
    }
    50% {
      box-shadow: inset 0 0 150px 5px var(--jump-color);
      opacity: 1;
    }
  }
  :global(.animate-jump-vignette) {
    animation: jump-vignette 0.8s infinite ease-in-out;
  }

  @keyframes label-in {
    from {
      opacity: 0.4;
      transform: translateY(2px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  :global(.animate-label-in) {
    animation: label-in 0.18s ease-out;
  }

  @media (prefers-reduced-motion: reduce) {
    :global(.animate-turn-glow),
    :global(.animate-turn-pill),
    :global(.animate-jump-cta),
    :global(.animate-jump-vignette),
    :global(.animate-label-in),
    :global(.animate-arrow-pulse) {
      animation: none;
    }
  }
</style>
