export type SeatPosition = "left" | "top" | "right";

export interface OpponentView {
    seat: number;
    role: string | null;
    suitedHand: string[];
    label: string;
    position: SeatPosition;
    is_turn: boolean;
    has_passed: boolean;
}